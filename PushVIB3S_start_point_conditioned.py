import torch
import torch.nn as nn
import torch.nn.functional as F

from bps import denormalize_torch

class PushVIB3S_start_point_conditioned(nn.Module):
    def __init__(self,
                 n_neurons=512,
                 in_bps=512,         # dimension of basis point set encoding
                 goalD=7,            #  6 + 1 for the goal
                 feature_dim=128,  # dimension for goal feature space
                 data_dir="processed_data/",
                 bps_dir="bps/",
                 use_directional_bps=False,
                 deterministic=False,
                 ):
        super(PushVIB3S_start_point_conditioned, self).__init__()
        
        print(f"[PushVIB3S_start_point_conditioned __init__] Received parameters: in_bps={in_bps}, use_directional_bps={use_directional_bps}")

        self.goalD = goalD
        self.data_dir = data_dir
        self.file_list = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cuda'
        self.max_dist = 0.0
        self.use_directional_bps = use_directional_bps
        self.deterministic = deterministic

        # Store the original in_bps value
        self.original_in_bps = in_bps
        
        # Adjust input dimension if using directional BPS
        if use_directional_bps:
            # For directional BPS: original dimension + 3D direction vectors
            self.in_bps = in_bps * 4  # 1 distance + 3 direction components per basis point
            print(f"[PushVIB3S_start_point_conditioned __init__] 'use_directional_bps' is True. Calculated self.in_bps = {self.in_bps}")
        else:
            self.in_bps = in_bps
            print(f"[PushVIB3S_start_point_conditioned __init__] 'use_directional_bps' is False. Calculated self.in_bps = {self.in_bps}")

        # ---------------------------
        # Goal Encoder: encodes the goal BPS into a latent space
        # ---------------------------
        print(f"[PushVIB3S_start_point_conditioned __init__] Initializing goal_encoder with self.in_bps = {self.in_bps}")
        self.goal_encoder = nn.Sequential(
            nn.Linear(self.in_bps, n_neurons),  # Changed from self.original_in_bps to n_neurons
            nn.BatchNorm1d(n_neurons),
            ResBlock(n_neurons, n_neurons),
            ResBlock(n_neurons, n_neurons)
        )
        # Projection head to create a target feature representation for reconstruction.
        self.goal_proj = nn.Linear(n_neurons, feature_dim)
        self.goal_mu = nn.Linear(n_neurons, self.goalD)
        self.goal_logvar = nn.Linear(n_neurons, self.goalD)

        # Goal Decoder: decodes z_goal into a reconstructed goal feature.
        self.goal_decoder = nn.Sequential(
            nn.Linear(self.goalD, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            ResBlock(64, 128),
            ResBlock(128, feature_dim)
        )

        print(f"[PushVIB3S_start_point_conditioned __init__] Initializing start_encoder with self.in_bps = {self.in_bps}")
        self.start_encoder = nn.Sequential(
            nn.Linear(self.in_bps, n_neurons),  # Changed from self.original_in_bps to n_neurons
            nn.BatchNorm1d(n_neurons),
            ResBlock(n_neurons, n_neurons),
            ResBlock(n_neurons, feature_dim)
        )

        self.start_point_proj = nn.Sequential(nn.Linear(3, feature_dim), nn.ReLU())

       # Prediction MLP (start feature + goal feature + start point feature)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim*3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Prediction heads
        self.end_point_head = nn.Linear(128, 3)  # xyz end point (normalized)
        print(f"[PushVIB3S_start_point_conditioned __init__] Model initialization complete.")

    def encode_goal(self, goal_bps):
        """Encode goal into latent space and project to a target feature representation."""
        hidden_goal = self.goal_encoder(goal_bps)  # [batch, n_neurons]
        # This projection serves as the reconstruction target for the goal.
        goal_feature_target = self.goal_proj(hidden_goal)
        mu = self.goal_mu(hidden_goal)
        logvar = torch.clamp(self.goal_logvar(hidden_goal), min=-20, max=2)
        return mu, logvar, goal_feature_target

    def project_goal(self, z_goal):
        """Project z_goal into a goal feature representation."""
        return self.goal_decoder(z_goal)

    def encode_start(self, start_bps):
        """Encode the start BPS into the start feature representation."""
        return self.start_encoder(start_bps)


    def forward(self, start_bps, goal_bps, start_point, x_mean=None, x_max=None):
        # Remove any extra dimensions
        if start_bps.dim() > 2:
            start_bps = start_bps.squeeze(1)
        if goal_bps.dim() > 2:
            goal_bps = goal_bps.squeeze(1)
                    
        # Check if we need to reshape the input when using directional BPS
        if self.use_directional_bps and start_bps.shape[1] != self.in_bps:
            # Reshape to match expected input size
            batch_size = start_bps.shape[0]
            start_bps = start_bps.reshape(batch_size, -1)
            goal_bps = goal_bps.reshape(batch_size, -1)
            
            # # If still not matching, pad with zeros
            # if start_bps.shape[1] < self.in_bps:
            #     pad_size = self.in_bps - start_bps.shape[1]
            #     start_bps = torch.cat([start_bps, torch.zeros(batch_size, pad_size, device=start_bps.device)], dim=1)
            #     goal_bps = torch.cat([goal_bps, torch.zeros(batch_size, pad_size, device=goal_bps.device)], dim=1)
        
        # Encode goal (get latent parameters and target features for reconstruction)
        goal_mu, goal_logvar, goal_feature_target = self.encode_goal(goal_bps)
        if self.deterministic:
            z_goal = goal_mu
        else:
            goal_std = goal_logvar.exp().pow(0.5)
            q_z_goal = torch.distributions.Normal(goal_mu, goal_std)
            z_goal = q_z_goal.rsample()

        # Decode goal features for reconstruction
        goal_feature = self.project_goal(z_goal)

        start_feature = self.encode_start(start_bps)
        start_point_feat = self.start_point_proj(start_point)  # [batch, feature_dim]

        # Concatenate the reconstructed goal with the start BPS
        combined_input = torch.cat([start_feature, goal_feature, start_point_feat], dim=1)  # [batch, feature_dim*3]

        # Pass through the MLP to predict the end point
        mlp_output = self.mlp(combined_input)
        end_point = self.end_point_head(mlp_output)  # [batch, 3], normalized

        # make sure x_mean and x_max are Double precision tensors
        if x_mean.dtype != torch.float32:
            x_mean = x_mean.to(torch.float32)
            x_max = x_max.to(torch.float32)
        
        # Remove the 2nd dimension from x_mean and x_max if they have more than two dimensions
        if x_mean.dim() > 2:
            x_mean = x_mean.squeeze(1)
        if x_max.dim() > 2:
            x_max = x_max.squeeze(1)

        end_point_final = denormalize_torch(end_point, x_mean, x_max)

        # Compute KL divergence
        if self.deterministic:
            kl_goal = torch.zeros((), device=goal_mu.device, dtype=goal_mu.dtype)
        else:
            kl_goal = -0.5 * torch.mean(torch.sum(1 + goal_logvar - goal_mu.pow(2) - goal_logvar.exp(), dim=1))

        return {
                    "end_point": end_point_final,
                    "normalized_end_point": end_point,
                    "goal_mu": goal_mu,
                    "goal_logvar": goal_logvar,
                    "kl_goal": kl_goal,
                    "latent_goal": z_goal,
                }

    def sample_from_posterior(self, start_bps, goal_bps, start_point, x_mean, x_max, num_samples=100):
        """Sample end points from the posterior distribution q(z_goal | goal_bps), vectorized.

        Encodes the goal once, draws num_samples latent codes via the reparameterization trick,
        then processes all samples in a single batched forward pass through the decoder and MLP.

        Args:
            start_bps (torch.Tensor): [B, in_bps] start BPS encoding
            goal_bps (torch.Tensor): [B, in_bps] goal BPS encoding
            start_point (torch.Tensor): [B, 3] normalized start point
            x_mean (torch.Tensor): [B, 3] normalization mean
            x_max (torch.Tensor): [B, 1] normalization max norm
            num_samples (int): number of posterior samples to draw per batch item

        Returns:
            torch.Tensor: [B, num_samples, 3] denormalized end points
        """
        if start_bps.dim() > 2:
            start_bps = start_bps.squeeze(1)
        if goal_bps.dim() > 2:
            goal_bps = goal_bps.squeeze(1)

        if x_mean.dtype != torch.float32:
            x_mean = x_mean.to(torch.float32)
            x_max = x_max.to(torch.float32)
        if x_mean.dim() > 2:
            x_mean = x_mean.squeeze(1)
        if x_max.dim() > 2:
            x_max = x_max.squeeze(1)

        B = start_bps.shape[0]

        goal_mu, goal_logvar, _ = self.encode_goal(goal_bps)        # (B, goalD)
        start_feature = self.encode_start(start_bps)                # (B, feature_dim)
        start_point_feat = self.start_point_proj(start_point)       # (B, feature_dim)

        # Sample num_samples z_goal vectors per batch item: (B, num_samples, goalD)
        goal_std = goal_logvar.exp().pow(0.5)
        eps = torch.randn(B, num_samples, self.goalD, device=goal_mu.device, dtype=goal_mu.dtype)
        z_goals = goal_mu.unsqueeze(1) + goal_std.unsqueeze(1) * eps  # (B, num_samples, goalD)

        # Flatten to (B * num_samples, ...) for a single batched pass
        z_goals_flat = z_goals.view(B * num_samples, self.goalD)
        goal_features_flat = self.project_goal(z_goals_flat)                      # (B*S, feature_dim)
        start_feature_flat = start_feature.repeat_interleave(num_samples, dim=0)  # (B*S, feature_dim)
        start_point_flat   = start_point_feat.repeat_interleave(num_samples, dim=0)

        combined = torch.cat([start_feature_flat, goal_features_flat, start_point_flat], dim=1)
        end_points_norm = self.end_point_head(self.mlp(combined))   # (B*S, 3)

        x_mean_flat = x_mean.repeat_interleave(num_samples, dim=0)  # (B*S, 3)
        x_max_flat  = x_max.repeat_interleave(num_samples, dim=0)   # (B*S, 1)
        end_points_world = denormalize_torch(end_points_norm, x_mean_flat, x_max_flat)

        return end_points_world.view(B, num_samples, 3)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, return_both=False):
        identity = self.shortcut(x)
        out = self.linear1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = F.relu(out + identity)
        if return_both:
            return out
        return out





