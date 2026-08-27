import torch
import torch.nn as nn
import torch_scatter
from ptv3.model import PointTransformerV3


class PTV3ObservationEncoder(nn.Module):
    def __init__(self, latent_dim=1024, grid_size=0.001, order="z", shuffle_orders=False, enable_flash=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.backbone = PointTransformerV3(
            in_channels=3,
            order=order,
            shuffle_orders=shuffle_orders,
            enable_flash=enable_flash,
            cls_mode=True,
        )
        enc_out_dim = self.backbone.enc_channels[-1]
        self.global_proj = nn.Sequential(nn.Linear(enc_out_dim, latent_dim), nn.ReLU())

    def build_ptv3_input(self, observations):
        if observations.ndim != 4:
            raise ValueError(f"observations must be (B,1,N,3), got {tuple(observations.shape)}")
        B, K, N, C = observations.shape
        if K != 1 or C != 3:
            raise ValueError(f"observations must be (B,1,N,3), got {tuple(observations.shape)}")

        # Encode a single point cloud
        coord = observations[:, 0] # Take a look at Yanxi's original code when wanting to add additional point clouds

        device = observations.device

        return {
            "coord": coord.reshape(B * N, 3), # B * num_point_clouds (1 in this case) * num_points_per_pc, 3
            "feat": coord.reshape(B * N, 3),
            "batch": torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N),
            "grid_size": self.grid_size,
        }

    def forward(self, observations):
        B = observations.shape[0]
        data_dict = self.build_ptv3_input(observations)
        enc_summary, _ = self.backbone(data_dict, return_endpoints=True)
        enc_global = torch_scatter.scatter_mean(
            enc_summary["feat"], enc_summary["batch"], dim=0, dim_size=B
        )
        return self.global_proj(enc_global)