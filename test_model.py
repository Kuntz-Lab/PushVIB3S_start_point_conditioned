import torch
import numpy as np
from PullVIBES_ptv3 import PullVIB3S
from dataset import PushDataset
import torch.nn as nn
import random
import csv

from pull_utils import set_seed, save_weight_comparison_boxplot, denormalize_torch


def load_weights(model, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint.state_dict())



def sample_and_visualize(model, start_pc_normalized, x_mean, x_max, n_samples=1000):
    # Encode the start point cloud into its feature vector
    start_feature = model.encode_start(start_pc_normalized)

    # Sample directly from a standard normal distribution in the goal latent space
    std = 1.0
    z_goals = torch.randn(n_samples, model.goalD).to(start_pc_normalized.device) * std
    
    all_start_points = []
    all_displacements = []

    all_start_points_normalized = []
    all_displacements_normalized = []

    mean = 0
    
    for z_goal in z_goals:
        # Decode the goal latent vector to get goal features
        goal_recon = model.decode_goal(z_goal.unsqueeze(0))

        mean += goal_recon.mean()
        # Concatenate start features with decoded goal features
        combined_input = torch.cat([start_feature, goal_recon], dim=1)
        
        # Pass through MLP and prediction heads
        mlp_output = model.mlp(combined_input)
        start_point_normalized = model.start_point_head(mlp_output)
        displacement_normalized = model.displacement_head(mlp_output)

        # denormalize start point and displacement
        start_point = denormalize_torch(start_point_normalized, x_mean, x_max)
        displacement = displacement_normalized * x_max
        
        all_start_points.append(start_point.cpu().numpy())
        all_displacements.append(displacement.cpu().numpy())

        all_start_points_normalized.append(start_point_normalized.cpu().numpy())
        all_displacements_normalized.append(displacement_normalized.cpu().numpy())

    # Combine all predictions into a single dictionary with numpy arrays
    predictions = {
        'start_point': np.vstack(all_start_points),
        'displacement': np.vstack(all_displacements),
        'normalized_start_point': np.vstack(all_start_points_normalized),
        'normalized_displacement': np.vstack(all_displacements_normalized)
    }

    # print(mean / n_samples)

    return predictions


def sample_and_visualize_posterior(model, start_pc_normalized, goal_pc_normalized, x_mean, x_max, n_samples=1000):
    # Get start features from the start encoder
    start_feature = model.encode_start(start_pc_normalized)

    # Get the posterior distribution parameters from encode_goal
    goal_mu, goal_logvar = model.encode_goal(goal_pc_normalized)

    print("mean:", goal_mu.mean())
    print("std:", goal_logvar.exp().pow(0.5).mean())
    
    all_start_points = []
    all_displacements = []
    all_start_points_normalized = []
    all_displacements_normalized = []

    z_goal_posterior = []

    mean = 0
    
    # Sample n_samples times from the posterior distribution
    for _ in range(n_samples):
        # Sample from the posterior N(mu, std)
        z_goal = goal_mu + torch.exp(0.5 * goal_logvar) * torch.randn_like(goal_mu)
        
        # Decode the goal latent vector to get goal features
        goal_recon = model.decode_goal(z_goal)
        
        # Concatenate start features with decoded goal features
        combined_input = torch.cat([start_feature, goal_recon], dim=1)
        
        # Pass through MLP and prediction heads
        mlp_output = model.mlp(combined_input)
        start_point_normalized = model.start_point_head(mlp_output)
        displacement_normalized = model.displacement_head(mlp_output)

        # Denormalize start point and displacement
        start_point = denormalize_torch(start_point_normalized, x_mean, x_max)
        displacement = displacement_normalized * x_max
        
        all_start_points.append(start_point.cpu().numpy())
        all_displacements.append(displacement.cpu().numpy())
        all_start_points_normalized.append(start_point_normalized.cpu().numpy())
        all_displacements_normalized.append(displacement_normalized.cpu().numpy())
        z_goal_posterior.append(z_goal.cpu().numpy())
        mean += goal_recon.mean()

    # Combine all predictions into a single dictionary with numpy arrays
    predictions = {
        'start_point': np.vstack(all_start_points),
        'displacement': np.vstack(all_displacements),
        'normalized_start_point': np.vstack(all_start_points_normalized),
        'normalized_displacement': np.vstack(all_displacements_normalized)
    }
    z_goal_posterior = np.vstack(z_goal_posterior)

    print(mean / n_samples)

    return predictions


def compute_variance(predictions):
    start_points_normalized = predictions['start_point'] * 1000  # convert to mm
    displacements_normalized = predictions['displacement'] * 1000  # convert to mm

    # take norm of displacements
    displacements_normalized = np.linalg.norm(displacements_normalized, axis=1, keepdims=True)
    
    all = np.concatenate([start_points_normalized, displacements_normalized], axis=1) # shape (n_samples, 6)

    cov_matrix_np = np.cov(all.T, bias=True)

    overall_variance = np.trace(cov_matrix_np)

    return overall_variance, cov_matrix_np

def compare_goal_features(model, z_prior, z_posterior):
    goal_features_prior = model.decode_goal(z_prior)
    goal_features_posterior = model.decode_goal(z_posterior)
    print(f"Feature difference: {torch.norm(goal_features_prior - goal_features_posterior).item()}")

def evaluate_model(
    dataset,
    model,
    device,
    evaluate_prior=False,
    visualize_prior=False,
    visualize_posterior=False,
    ):
    start_point_error_sum = 0.0
    displacement_error_sum = 0.0

    start_point_rmse_sum = 0.0
    displacement_rmse_sum = 0.0

    start_point_errors = []
    displacement_errors = []

    prior_start_point_variances = []
    prior_displacement_variances = []
    missing_ground_truth = False

    for i in range(len(dataset)):
        data = dataset[i]

        start_pc_normalized = data['start_pc'].unsqueeze(0).to(device)
        goal_pc_normalized = data['goal_pc'].unsqueeze(0).to(device)

        gt_start = None
        gt_displacement = None
        if 'start_point' in data and 'displacement' in data:
            gt_start = data['start_point']
            gt_displacement = data['displacement']
        else:
            missing_ground_truth = True
        
        model.deterministic = True  # Use mean prediction for evaluation
        data['x_mean'] = torch.tensor(data['x_mean'], dtype=torch.float32).to(device)
        data['x_max'] = torch.tensor(data['x_max'], dtype=torch.float32).to(device)
        outputs = model(start_pc_normalized, goal_pc_normalized, x_mean=data['x_mean'], x_max=data['x_max'])
        single_pred = {
            'start_point': outputs['start_point'].cpu().numpy(),
            'displacement': outputs['displacement'].cpu().numpy()
        }
        model.deterministic = False  # Switch back to stochastic for sampling and visualization

        if visualize_prior and not evaluate_prior:
            print("Must set evaluate_prior=True to visualize prior samples. Skipping prior visualization.")

        if evaluate_prior:
            _predictions = sample_and_visualize(
                model,
                start_pc_normalized,
                data['x_mean'],
                data['x_max'],
                n_samples=200,
            )

            prior_start_points = _predictions['start_point']
            prior_displacements = _predictions['displacement']

            start_centroid = np.mean(prior_start_points, axis=0)
            disp_centroid = np.mean(prior_displacements, axis=0)

            start_sq_dists = np.sum((prior_start_points - start_centroid) ** 2, axis=1)
            disp_sq_dists = np.sum((prior_displacements - disp_centroid) ** 2, axis=1)

            prior_start_point_variances.append(float(np.mean(start_sq_dists)))
            prior_displacement_variances.append(float(np.mean(disp_sq_dists)))

        if visualize_posterior:
            predictions = sample_and_visualize_posterior(
                model,
                start_pc_normalized,
                goal_pc_normalized,
                data['x_mean'],
                data['x_max'],
                n_samples=500,
            )

            overall_variance, _covariance_matrix = compute_variance(predictions)
            print(f"Overall variance (trace of covariance): {overall_variance}")

        if missing_ground_truth:
            continue

        start_point_error = np.linalg.norm(single_pred['start_point'] - gt_start)
        displacement_error = np.linalg.norm(single_pred['displacement'] - gt_displacement)

        start_point_errors.append(start_point_error)
        displacement_errors.append(displacement_error)

        start_point_error_sum += start_point_error
        displacement_error_sum += displacement_error

        start_point_error_squared = start_point_error ** 2
        displacement_error_squared = displacement_error ** 2
        start_point_rmse_sum += start_point_error_squared
        displacement_rmse_sum += displacement_error_squared

    if missing_ground_truth:
        return None, None

    avg_start_point_error = start_point_error_sum / len(dataset)
    avg_displacement_error = displacement_error_sum / len(dataset)
    rmse_start_point = (start_point_rmse_sum / len(dataset)) ** 0.5
    rmse_displacement = (displacement_rmse_sum / len(dataset)) ** 0.5

    start_point_q25, start_point_median, start_point_q75 = np.percentile(start_point_errors, [25, 50, 75])
    displacement_q25, displacement_median, displacement_q75 = np.percentile(displacement_errors, [25, 50, 75])

    avg_prior_start_point_variance = float(np.mean(prior_start_point_variances))
    avg_prior_displacement_variance = float(np.mean(prior_displacement_variances))

    stats = {
        'avg_start_point_error': float(avg_start_point_error),
        'avg_displacement_error': float(avg_displacement_error),
        'rmse_start_point': float(rmse_start_point),
        'rmse_displacement': float(rmse_displacement),
        'start_point_median': float(start_point_median),
        'start_point_q25': float(start_point_q25),
        'start_point_q75': float(start_point_q75),
        'start_point_min': float(np.min(start_point_errors)),
        'start_point_max': float(np.max(start_point_errors)),
        'displacement_median': float(displacement_median),
        'displacement_q25': float(displacement_q25),
        'displacement_q75': float(displacement_q75),
        'displacement_min': float(np.min(displacement_errors)),
        'displacement_max': float(np.max(displacement_errors)),
        'prior_start_point_avg_variance': avg_prior_start_point_variance,
        'prior_displacement_avg_variance': avg_prior_displacement_variance,
        'num_samples': len(dataset),
    }

    plot_data = {
        'start_point_errors': [float(error) for error in start_point_errors],
        'displacement_errors': [float(error) for error in displacement_errors],
    }

    return stats, plot_data

def test_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Add PushNetGoalVAE to safe globals
    torch.serialization.add_safe_globals([PullVIB3S])

    # Load model
    model = PullVIB3S(
        n_neurons=512,
        goalD=7,
        feature_dim=128,
    ).to(device)

    vibes_models_to_test = [
        # {"name": "trained in sim only", "weight_path": "/home/britton/PushVIBES/weights/oriented/new_old_data_mix_checkpoint_noisy_115_0.000462.pt"},
        # {"name": "fine-tuned, no sim", "weight_path": "/home/britton/PushVIBES/checkpoints/pretraining_local_20260225_130439/best_model_weights.pt"},
        # {"name": "fine-tuned, 50 sim per epoch", "weight_path": "/home/britton/PushVIBES/checkpoints/pretraining_local_20260224_172027/best_model_weights.pt"},
        # {"name": "fine-tuned, 200 sim per epoch", "weight_path": "/home/britton/PushVIBES/checkpoints/pretraining_local_20260224_111901/best_model_weights.pt"},
        # {"name": "fine-tuned, 410 sim per epoch, KL divergence weight 1e-4", "weight_path": "/home/britton/PushVIBES/checkpoints/pretraining_local_20260223_181954_KLe4/checkpoint_noisy_193_0.325406.pt"},
        # {"name": "trained on real only", "weight_path": "/home/britton/PushVIBES/checkpoints/local_20260226_172711/best_model_weights.pt"},
        # {"name": "fine-tuned, 410 sim per epoch", "weight_path": "/home/britton/PushVIBES/checkpoints/finetune_local_20260302_113913/best_model_weights.pt"},
        # {"name": "fine-tuned, 300 sim per epoch", "weight_path": "/home/britton/PushVIBES/checkpoints/finetune_local_20260305_152236/best_model_weights.pt"},
        # {"name": "fine-tuned, 561 sim/real", "weight_path": "/home/britton/PushVIBES/checkpoints/561real_561sim_lr8e5_finetunelocal_20260320_192840/checkpoint_133_0.347904.pt"},
        {"name": "cotrained, 561 sim/real, 1e-4", "weight_path": "/home/britton/PushVIBES/checkpoints/561real_561sim_lr1e4_cotrainlocal_20260323_131410/best_model_weights.pt"},
        {"name": "deterministic, cotrained, 561 sim/real, 1e-4", "weight_path": "/home/britton/PushVIBES/checkpoints/deterministic/deterministiclocal_20260403_165056/checkpoint_321_0.334218.pt"},
        # {"name": "deterministic, finetuned", "weight_path": "checkpoints/deterministic/deterministic_finetunelocal_20260407_184340/checkpoint_196_0.356486.pt"}

    ]

    results_rows = []
    plot_rows = []

    real_world_test_dataset = PushDataset(
        data_dir="./data/all_good_realsense_data/test",
        deterministic_farthest_point_sampling=True,
    )

    dataset = PushDataset(
        data_dir='./data/aug7_2025_new_data/processed_data_noisy_object_frame_with_old_and_new_data/',
        deterministic_farthest_point_sampling=True
    )
    
    # Use 90% for training, 10% for validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    print("Train size: ", train_size)
    print("Val size: ", val_size)
    
    _train_dataset, sim_test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    
    # randomly subsample sim test dataset to sim_test_num_samples samples for faster evaluation
    sim_test_num_samples = 100 # 1000
    sim_test_dataset = torch.utils.data.Subset(
            sim_test_dataset,
            random.sample(range(len(sim_test_dataset)), min(sim_test_num_samples, len(sim_test_dataset)))
        )

    for model_to_test in vibes_models_to_test:
        print(f"Testing model: {model_to_test['name']} with weights from {model_to_test['weight_path']}")
        try:
            load_weights(model, model_to_test["weight_path"], device)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return          

        model.eval()


        with torch.no_grad():
            real_stats, real_plot_data = evaluate_model(
                real_world_test_dataset,
                model,
                device,
                evaluate_prior=False,
                visualize_prior=False,
                visualize_posterior=False,
            )

            print(f"Avg. error, start point, real-world test set: {round(real_stats['avg_start_point_error'], 4)} meters")
            print(f"Avg. error, displacement, real-world test set: {round(real_stats['avg_displacement_error'], 4)} meters")
            # print(f"RMSE, start point, real-world test set: {round(real_stats['rmse_start_point'], 4)} meters")
            # print(f"RMSE, displacement, real-world test set: {round(real_stats['rmse_displacement'], 4)} meters")

            sim_stats, sim_plot_data = evaluate_model(
                sim_test_dataset,
                model,
                device,
                evaluate_prior=False,
                visualize_prior=False,
                visualize_posterior=False,
            )

            print(f"Avg. error, start point, sim test set: {round(sim_stats['avg_start_point_error'], 4)} meters")
            print(f"Avg. error, displacement, sim test set: {round(sim_stats['avg_displacement_error'], 4)} meters")
            # print(f"RMSE, start point, sim test set: {round(sim_stats['rmse_start_point'], 4)} meters")
            # print(f"RMSE, displacement, sim test set: {round(sim_stats['rmse_displacement'], 4)} meters")

            row = {
                'model_name': model_to_test['name'],
                'weights_path': model_to_test['weight_path'],
            }
            row.update({f"real_{key}": value for key, value in real_stats.items()})
            row.update({f"sim_{key}": value for key, value in sim_stats.items()})
            results_rows.append(row)

            plot_rows.append({
                'model_name': model_to_test['name'],
                'real_start_point_errors': real_plot_data['start_point_errors'],
                'real_displacement_errors': real_plot_data['displacement_errors'],
                'sim_start_point_errors': sim_plot_data['start_point_errors'],
                'sim_displacement_errors': sim_plot_data['displacement_errors'],
            })

    if results_rows:
        csv_path = "model_evaluation_stats.csv"
        fieldnames = []
        for row in results_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"Saved evaluation stats to {csv_path}")

        plot_path = "model_weight_comparison_boxplot.png"
        save_weight_comparison_boxplot(plot_rows, plot_path)
        print(f"Saved weight comparison box plot to {plot_path}")
            


            


def test_goal_importance(model, start_pc_normalized, goal_pc_normalized, device):
    # Get start features
    start_feature = model.encode_start(start_pc_normalized)
    
    # Case 1: Use zero goal features
    zero_goal = torch.zeros((1, model.mlp[0].in_features - start_feature.size(1))).to(device)
    combined_1 = torch.cat([start_feature, zero_goal], dim=1)
    
    # Case 2: Use random goal features
    random_goal = torch.randn((1, model.mlp[0].in_features - start_feature.size(1))).to(device)
    combined_2 = torch.cat([start_feature, random_goal], dim=1)
    
    # Get predictions for both cases
    mlp_output_1 = model.mlp(combined_1)
    pred_1_start = model.start_point_head(mlp_output_1)
    pred_1_disp = model.displacement_head(mlp_output_1)
    
    mlp_output_2 = model.mlp(combined_2)
    pred_2_start = model.start_point_head(mlp_output_2)
    pred_2_disp = model.displacement_head(mlp_output_2)
    
    # Compare predictions
    start_diff = torch.norm(pred_1_start - pred_2_start)
    disp_diff = torch.norm(pred_1_disp - pred_2_disp)
    print(f"Difference between predictions with zero vs random goal:")
    print(f"Start point diff: {start_diff.item():.4f}")
    print(f"Displacement diff: {disp_diff.item():.4f}")

if __name__ == "__main__":
    set_seed(333)
    test_model()
