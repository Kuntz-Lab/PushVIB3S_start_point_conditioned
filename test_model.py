import torch
import numpy as np
from PushVIB3S_start_point_conditioned import (
    PushVIB3S_start_point_conditioned,
    STRATEGY_END_POINT,
    ENCODER_BPS,
    ENCODER_PTV3,
)
from dataset import PushDataset
import torch.nn as nn
import random
import csv

from utils import set_seed, save_weight_comparison_boxplot


def _model_inputs(data, encoder_type):
    """Pick the (start, goal) tensors PushDataset produced for encoder_type: BPS vectors
    for "bps", raw normalized point clouds for "ptv3". Mirrors PushVibesTrainer._model_inputs
    in train.py."""
    if encoder_type == ENCODER_PTV3:
        return data['start_pc'], data['goal_pc']
    return data['start_bps'], data['goal_bps']


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


def sample_and_visualize(model, start_input, start_point, x_mean, x_max, n_samples=1000):
    """Sample end points from the prior p(z_goal) = N(0, I), vectorized."""
    end_points_world = model.sample_from_prior(
        start_input, start_point, x_mean, x_max, num_samples=n_samples
    )  # (B, n_samples, 3)

    predictions = {
        'end_point': end_points_world.cpu().numpy(),
    }

    return predictions


def sample_and_visualize_posterior(model, start_input, goal_input, start_point, x_mean, x_max, n_samples=1000):
    """Sample end points from the posterior q(z_goal | goal_input), vectorized."""
    goal_mu, goal_logvar, _ = model.encode_goal(goal_input)
    print("mean:", goal_mu.mean().item())
    print("std:", goal_logvar.exp().pow(0.5).mean().item())

    end_points_world = model.sample_from_posterior(
        start_input, goal_input, start_point, x_mean, x_max, num_samples=n_samples
    )  # (B, n_samples, 3)

    predictions = {
        'end_point': end_points_world.cpu().numpy(),
    }

    return predictions


def compute_variance(predictions):
    end_points = predictions['end_point'].reshape(-1, 3) * 1000  # convert to mm

    cov_matrix_np = np.cov(end_points.T, bias=True)
    overall_variance = np.trace(cov_matrix_np)

    return overall_variance, cov_matrix_np


def compare_goal_features(model, z_prior, z_posterior):
    goal_features_prior = model.project_goal(z_prior)
    goal_features_posterior = model.project_goal(z_posterior)
    print(f"Feature difference: {torch.norm(goal_features_prior - goal_features_posterior).item()}")


def evaluate_model(
    dataset,
    model,
    device,
    evaluate_prior=False,
    visualize_posterior=False,
    ):
    end_point_error_sum = 0.0
    end_point_rmse_sum = 0.0
    end_point_errors = []

    prior_end_point_variances = []
    missing_ground_truth = False

    for i in range(len(dataset)):
        data = dataset[i]

        start_input, goal_input = _model_inputs(data, model.encoder_type)
        start_input = start_input.unsqueeze(0).to(device)
        goal_input = goal_input.unsqueeze(0).to(device)
        start_point_normalized = torch.tensor(
            data['start_point_normalized'], dtype=torch.float32
        ).to(device).view(1, 3)
        x_mean = torch.tensor(data['x_mean'], dtype=torch.float32).to(device)
        x_max = torch.tensor(data['x_max'], dtype=torch.float32).to(device)

        gt_end_point = None
        if 'end_point' in data:
            gt_end_point = data['end_point']
        else:
            missing_ground_truth = True

        model.deterministic = True  # Use mean prediction for evaluation
        outputs = model(start_input, goal_input, start_point_normalized, x_mean=x_mean, x_max=x_max)
        single_pred = {'end_point': outputs['end_point'].cpu().numpy()}
        model.deterministic = False  # Switch back to stochastic for sampling and visualization

        if evaluate_prior:
            _predictions = sample_and_visualize(
                model,
                start_input,
                start_point_normalized,
                x_mean,
                x_max,
                n_samples=200,
            )

            prior_end_points = _predictions['end_point'].reshape(-1, 3)
            end_point_centroid = np.mean(prior_end_points, axis=0)
            end_point_sq_dists = np.sum((prior_end_points - end_point_centroid) ** 2, axis=1)
            prior_end_point_variances.append(float(np.mean(end_point_sq_dists)))

        if visualize_posterior:
            predictions = sample_and_visualize_posterior(
                model,
                start_input,
                goal_input,
                start_point_normalized,
                x_mean,
                x_max,
                n_samples=500,
            )

            overall_variance, _covariance_matrix = compute_variance(predictions)
            print(f"Overall variance (trace of covariance): {overall_variance}")

        if missing_ground_truth:
            continue

        end_point_error = np.linalg.norm(single_pred['end_point'] - gt_end_point)
        end_point_errors.append(end_point_error)
        end_point_error_sum += end_point_error
        end_point_rmse_sum += end_point_error ** 2

    if missing_ground_truth:
        return None, None

    avg_end_point_error = end_point_error_sum / len(dataset)
    rmse_end_point = (end_point_rmse_sum / len(dataset)) ** 0.5

    end_point_q25, end_point_median, end_point_q75 = np.percentile(end_point_errors, [25, 50, 75])
    avg_prior_end_point_variance = float(np.mean(prior_end_point_variances)) if prior_end_point_variances else float('nan')

    stats = {
        'avg_end_point_error': float(avg_end_point_error),
        'rmse_end_point': float(rmse_end_point),
        'end_point_median': float(end_point_median),
        'end_point_q25': float(end_point_q25),
        'end_point_q75': float(end_point_q75),
        'end_point_min': float(np.min(end_point_errors)),
        'end_point_max': float(np.max(end_point_errors)),
        'prior_end_point_avg_variance': avg_prior_end_point_variance,
        'num_samples': len(dataset),
    }

    plot_data = {
        'end_point_errors': [float(error) for error in end_point_errors],
    }

    return stats, plot_data


def test_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_neurons = 512
    in_bps = 128
    goalD = 7
    feature_dim = 128
    use_directional_bps = True

    # Add PushVIB3S_start_point_conditioned to safe globals
    torch.serialization.add_safe_globals([PushVIB3S_start_point_conditioned])

    vibes_models_to_test = [
        # Fill in with checkpoints to evaluate. encoder_type/strategy must match what each
        # checkpoint was actually trained with -- both a model and (for "bps") a dataset are
        # built fresh per entry, so mismatched checkpoints/entries fail loudly at
        # load_state_dict(strict=True) instead of silently scoring through the wrong heads.
        # {"name": "my run", "weight_path": "checkpoints/<run_name>/best_model_weights.pt",
        #  "encoder_type": ENCODER_BPS, "strategy": STRATEGY_END_POINT},
    ]

    if not vibes_models_to_test:
        print("No models listed in vibes_models_to_test — add entries with a name, weight_path, "
              "encoder_type, and strategy before running.")
        return

    results_rows = []
    plot_rows = []

    # Real-world/sim test datasets, built once per distinct encoder_type actually needed
    # ("bps" datasets require the BPS basis file; "ptv3" ones don't).
    datasets_by_encoder: dict = {}

    def _get_datasets(encoder_type):
        if encoder_type not in datasets_by_encoder:
            real_world_test_dataset = PushDataset(
                data_dir="./data/all_good_realsense_data/test",
                use_directional_bps=use_directional_bps,
                deterministic_farthest_point_sampling=True,
                encoder_type=encoder_type,
            )

            dataset = PushDataset(
                data_dir='./data/aug7_2025_new_data/processed_data_noisy_object_frame_with_old_and_new_data/',
                use_directional_bps=use_directional_bps,
                deterministic_farthest_point_sampling=True,
                encoder_type=encoder_type,
            )

            # Use 90% for training, 10% for validation
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size

            print(f"[{encoder_type}] Train size: ", train_size)
            print(f"[{encoder_type}] Val size: ", val_size)

            _train_dataset, sim_test_dataset = torch.utils.data.random_split(
                    dataset, [train_size, val_size]
                )

            # randomly subsample sim test dataset to sim_test_num_samples samples for faster evaluation
            sim_test_num_samples = 100 # 1000
            sim_test_dataset = torch.utils.data.Subset(
                    sim_test_dataset,
                    random.sample(range(len(sim_test_dataset)), min(sim_test_num_samples, len(sim_test_dataset)))
                )

            datasets_by_encoder[encoder_type] = (real_world_test_dataset, sim_test_dataset)

        return datasets_by_encoder[encoder_type]

    for model_to_test in vibes_models_to_test:
        encoder_type = model_to_test.get("encoder_type", ENCODER_BPS)
        strategy = model_to_test.get("strategy", STRATEGY_END_POINT)

        print(f"Testing model: {model_to_test['name']} (encoder_type={encoder_type}, strategy={strategy}) "
              f"with weights from {model_to_test['weight_path']}")

        model = PushVIB3S_start_point_conditioned(
            n_neurons=n_neurons,
            in_bps=in_bps,
            goalD=goalD,
            feature_dim=feature_dim,
            use_directional_bps=use_directional_bps,
            encoder_type=encoder_type,
            strategy=strategy,
        ).to(device)

        try:
            load_weights(model, model_to_test["weight_path"], device)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return

        model.eval()

        real_world_test_dataset, sim_test_dataset = _get_datasets(encoder_type)

        with torch.no_grad():
            real_stats, real_plot_data = evaluate_model(
                real_world_test_dataset,
                model,
                device,
                evaluate_prior=False,
                visualize_posterior=False,
            )

            print(f"Avg. error, end point, real-world test set: {round(real_stats['avg_end_point_error'], 4)} meters")

            sim_stats, sim_plot_data = evaluate_model(
                sim_test_dataset,
                model,
                device,
                evaluate_prior=False,
                visualize_posterior=False,
            )

            print(f"Avg. error, end point, sim test set: {round(sim_stats['avg_end_point_error'], 4)} meters")

            row = {
                'model_name': model_to_test['name'],
                'weights_path': model_to_test['weight_path'],
            }
            row.update({f"real_{key}": value for key, value in real_stats.items()})
            row.update({f"sim_{key}": value for key, value in sim_stats.items()})
            results_rows.append(row)

            plot_rows.append({
                'model_name': model_to_test['name'],
                'real_end_point_errors': real_plot_data['end_point_errors'],
                'sim_end_point_errors': sim_plot_data['end_point_errors'],
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


def test_goal_importance(model, start_input, start_point, device):
    """Compare predicted end points when the goal feature is zeroed out vs. randomized,
    for a fixed start input (BPS vector or point cloud) and start point."""
    feature_dim = model.start_point_proj[0].out_features

    start_feature = model.encode_start(start_input)
    start_point_feat = model.start_point_proj(start_point)

    zero_goal = torch.zeros((1, feature_dim)).to(device)
    combined_1 = torch.cat([start_feature, zero_goal, start_point_feat], dim=1)

    random_goal = torch.randn((1, feature_dim)).to(device)
    combined_2 = torch.cat([start_feature, random_goal, start_point_feat], dim=1)

    pred_1 = model.end_point_head(model.mlp(combined_1))
    pred_2 = model.end_point_head(model.mlp(combined_2))

    end_point_diff = torch.norm(pred_1 - pred_2)
    print(f"Difference between predictions with zero vs random goal:")
    print(f"End point diff: {end_point_diff.item():.4f}")


if __name__ == "__main__":
    set_seed(333)
    test_model()
