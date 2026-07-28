import torch
import numpy as np

from PushVIB3S_start_point_conditioned import PushVIB3S_start_point_conditioned
from dataset import PushDataset
from test_model import load_weights
from utils import set_seed, visualize_action_prediction


if __name__ == "__main__":
    seed = 333
    set_seed(seed)

    dataset_path = "/home/britton/PushVIBES/data/all_good_realsense_data/test"

    # weights_path = "/home/britton/PushVIB3S_start_point_conditioned/checkpoints/realsense_bg0.08local_20260724_142009/checkpoint_200_0.035985.pt" #
    # weights_path = "/home/britton/PushVIB3S_start_point_conditioned/checkpoints/realsense_bg0.01local_20260724_135500/checkpoint_-1_0.031510.pt"
    # weights_path = "/home/britton/PushVIB3S_start_point_conditioned/checkpoints/realsense_bg0.01local_20260724_135500/latest_model_weights.pt"
    # weights_path = "/home/britton/PushVIB3S_start_point_conditioned/checkpoints/realsense_bg0.005local_20260724_153206/checkpoint_280_0.030861.pt"
    # weights_path = "/home/britton/PushVIB3S_start_point_conditioned/checkpoints/realsense_bg0.0005_weight1.5local_20260724_165655/checkpoint_592_0.027443.pt"
    weights_path = "/home/britton/PushVIB3S_start_point_conditioned/checkpoints/realsense_bg0.0005_weight1.5_w_no_diff_datalocal_20260727_165643/checkpoint_368_0.021833.pt"

    deterministic_farthest_point_sampling = True
    n_neurons = 512
    in_bps = 128
    goalD = 7
    feature_dim = 128
    use_directional_bps = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.serialization.add_safe_globals([PushVIB3S_start_point_conditioned])

    dataset = PushDataset(
        data_dir=dataset_path,
        use_directional_bps=use_directional_bps,
        deterministic_farthest_point_sampling=deterministic_farthest_point_sampling,
    )

    model = PushVIB3S_start_point_conditioned(
        n_neurons=n_neurons,
        in_bps=in_bps,
        goalD=goalD,
        feature_dim=feature_dim,
        use_directional_bps=use_directional_bps,
        deterministic=True,
    ).to(device)

    print(f"Testing weights: {weights_path}")
    print(f"Using dataset:   {dataset_path}")
    print(f"N samples:       {len(dataset)}")

    visualize_distribution = True
    n_distribution_samples = 100

    load_weights(model, weights_path, device)
    model.eval()
    model.deterministic = True

    end_point_errors = []

    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]

            start_bps   = data['start_bps'].unsqueeze(0).to(device)          # (1, in_bps)
            goal_bps    = data['goal_bps'].unsqueeze(0).to(device)           # (1, in_bps)
            start_point = torch.tensor(data['start_point_normalized'],
                                       dtype=torch.float32).to(device).view(1, 3)  # (1, 3)
            x_mean      = torch.tensor(data['x_mean'],
                                       dtype=torch.float32).to(device)
            x_max       = torch.tensor(data['x_max'],
                                       dtype=torch.float32).to(device)

            gt_end_point = data['end_point'].squeeze(0)                      # (3,)

            if visualize_distribution:
                sampled_prior = model.sample_from_prior(
                    start_bps, start_point,
                    x_mean=x_mean, x_max=x_max,
                    num_samples=n_distribution_samples,
                )  # (1, n_distribution_samples, 3)
                visualize_action_prediction(
                    start_point=data['start_point'].squeeze(0),
                    end_points=list(sampled_prior.squeeze(0).cpu().numpy()),
                    start_pc=data['start_pc_unnormalized'],
                    goal_pc=None,
                    gt_end_point=None,
                )

                sampled_posterior = model.sample_from_posterior(
                    start_bps, goal_bps, start_point=start_point,
                    x_mean=x_mean, x_max=x_max,
                    num_samples=n_distribution_samples,
                )  # (1, n_distribution_samples, 3)
                visualize_action_prediction(
                    start_point=data['start_point'].squeeze(0),
                    end_points=list(sampled_posterior.squeeze(0).cpu().numpy()),
                    start_pc=data['start_pc_unnormalized'],
                    goal_pc=data['goal_pc_unnormalized'],
                    gt_end_point=gt_end_point,
                )

            outputs = model(start_bps, goal_bps, start_point,
                            x_mean=x_mean, x_max=x_max)

            pred_end_point = outputs['end_point'].squeeze(0).cpu().numpy()   # (3,)

            visualize_action_prediction(
                start_point=data['start_point'].squeeze(0),
                end_points=[pred_end_point],
                start_pc=data['start_pc_unnormalized'],
                goal_pc=data['goal_pc_unnormalized'],
                gt_end_point=gt_end_point,
            )

            

            end_point_errors.append(np.linalg.norm(pred_end_point - gt_end_point))

    end_point_errors = np.array(end_point_errors)
    print(f"\nAvg. end point error:   {np.mean(end_point_errors):.4f} m")
    print(f"Median end point error: {np.median(end_point_errors):.4f} m")
    print(f"RMSE end point:         {np.sqrt(np.mean(end_point_errors ** 2)):.4f} m")
    print(f"Min / Max:              {np.min(end_point_errors):.4f} / {np.max(end_point_errors):.4f} m")
