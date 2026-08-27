"""Deploy a trained PushVIB3S_start_point_conditioned model without going through PushDataset.

This mirrors dataset.py's preprocessing pipeline (objectframeize -> downsample -> normalize ->
encode) but operates on a single start/goal point cloud pair supplied directly by the caller,
instead of piping pickle files through PushDataset/DataLoader.

Important: PushVIB3S_start_point_conditioned is *start-point conditioned* -- unlike the older
PushVIBES/PushDeterministic models, it does not predict where to push from. The caller must
supply `push_start_point`, the world-frame point on the object where the push begins (e.g. from
a grasp/contact planner), and the model predicts where that point ends up.
"""

from pathlib import Path
import pickle

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError

from bps import encode_pcd_with_bps, normalize
from PushVIB3S_start_point_conditioned import (
    PushVIB3S_start_point_conditioned,
    STRATEGY_END_POINT,
    ENCODER_BPS,
    ENCODER_PTV3,
)
from test_model import load_weights
from utils import (
    deobjectframeize_point,
    farthest_point_sampling,
    find_pickle_files,
    objectframeize_goal_pc,
    objectframeize_start_pc,
    visualize_action_prediction,
)

torch.serialization.add_safe_globals([PushVIB3S_start_point_conditioned])


def is_small_vector(direction, tol=1e-9):
    return np.linalg.norm(direction) <= tol


def compute_convex_hull_halfspaces(point_cloud):
    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3).")

    unique_points = np.unique(point_cloud, axis=0)
    if unique_points.shape[0] < 4:
        raise ValueError("Need at least 4 unique points for a 3D hull.")
    if np.linalg.matrix_rank(unique_points - unique_points.mean(axis=0)) < 3:
        raise ValueError("Point cloud must contain non-coplanar points.")

    hull = ConvexHull(unique_points)
    return hull.equations


def find_first_ray_exit_from_convex_hull(start_point, direction, hull_equations, tol=1e-9):
    start_point = np.asarray(start_point, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    hull_equations = np.asarray(hull_equations, dtype=np.float64)

    if start_point.shape != (3,):
        raise ValueError("Start point must have shape (3,).")
    if direction.shape != (3,):
        raise ValueError("Direction must have shape (3,).")
    if hull_equations.ndim != 2 or hull_equations.shape[1] != 4:
        raise ValueError("Hull equations must have shape (M, 4).")

    valid_t = []
    for equation in hull_equations:
        normal = equation[:3]
        offset = equation[3]
        denom = float(np.dot(normal, direction))
        if denom <= tol:
            continue

        numer = -float(np.dot(normal, start_point) + offset)
        t = numer / denom
        if t >= -tol:
            valid_t.append(max(0.0, t))

    if not valid_t:
        raise ValueError("No valid hull exit intersection found.")

    t_exit = min(valid_t)
    boundary_point = start_point + t_exit * direction
    return boundary_point


def compute_hull_exit_start_point(start_point_cloud, start_point, direction, tol=1e-9):
    """Used by DeployPushHeuristic to nudge its guessed start point onto the object surface."""
    start_point_cloud = np.asarray(start_point_cloud, dtype=np.float64)
    start_point = np.asarray(start_point, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)

    if start_point_cloud.ndim != 2 or start_point_cloud.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3).")
    if start_point.shape != (3,):
        raise ValueError("Start point must have shape (3,).")
    if direction.shape != (3,):
        raise ValueError("Direction must have shape (3,).")

    if is_small_vector(direction, tol=tol):
        return start_point

    try:
        hull_equations = compute_convex_hull_halfspaces(start_point_cloud)
        boundary_point = find_first_ray_exit_from_convex_hull(
            start_point,
            direction,
            hull_equations,
            tol=tol,
        )
    except (ValueError, QhullError, FloatingPointError):
        return start_point

    if not np.all(np.isfinite(boundary_point)):
        return start_point

    return boundary_point


def to_numpy_point_cloud(point_cloud):
    if torch.is_tensor(point_cloud):
        point_cloud = point_cloud.detach().cpu().numpy()

    point_cloud = np.asarray(point_cloud, dtype=np.float32)

    if point_cloud.ndim != 2:
        raise ValueError("Point cloud must have shape (N, 3).")
    if point_cloud.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3).")
    if point_cloud.shape[0] == 0:
        raise ValueError("Point cloud must be non-empty.")

    return point_cloud


def to_numpy_point(point):
    if torch.is_tensor(point):
        point = point.detach().cpu().numpy()
    point = np.asarray(point, dtype=np.float32).reshape(3)
    return point


def objectframeize_point(point, rotation_matrix, tissue_mean):
    """Forward counterpart of utils.deobjectframeize_point: world frame -> object frame."""
    return np.dot(point - tissue_mean, rotation_matrix)


def objectframeize_inputs(start_point_cloud, goal_point_cloud, push_start_point):
    start_pc_object_frame, rotation_matrix, tissue_mean = objectframeize_start_pc(
        start_point_cloud.copy()
    )
    goal_pc_object_frame = objectframeize_goal_pc(
        goal_point_cloud.copy(), rotation_matrix, tissue_mean
    )
    push_start_point_object_frame = objectframeize_point(
        push_start_point, rotation_matrix, tissue_mean
    )
    return (
        start_pc_object_frame,
        goal_pc_object_frame,
        push_start_point_object_frame,
        rotation_matrix,
        tissue_mean,
    )


def downsample_point_clouds(
    start_pc_obj,
    goal_pc_obj,
    num_points_per_pc,
    deterministic_farthest_point_sampling=True,
):
    use_internally_random_seed = not deterministic_farthest_point_sampling

    if start_pc_obj.shape[0] > num_points_per_pc:
        start_pc_obj = farthest_point_sampling(
            start_pc_obj,
            num_points_per_pc,
            use_internally_random_seed=use_internally_random_seed,
        )
    if goal_pc_obj.shape[0] > num_points_per_pc:
        goal_pc_obj = farthest_point_sampling(
            goal_pc_obj,
            num_points_per_pc,
            use_internally_random_seed=use_internally_random_seed,
        )

    return start_pc_obj, goal_pc_obj


def normalize_inputs(start_pc_obj, goal_pc_obj, push_start_point_obj, device):
    """Normalize start/goal point clouds onto the BPS unit sphere, and apply the same
    scalers to the push start point. Mirrors PushDataset.__getitem__'s normalization step."""
    start_batch = np.expand_dims(np.copy(start_pc_obj), axis=0)
    goal_batch = np.expand_dims(np.copy(goal_pc_obj), axis=0)

    start_normalized, x_mean, x_max = normalize(start_batch, return_scalers=True)
    goal_normalized, _, _ = normalize(
        goal_batch,
        known_scalers=True,
        x_mean=x_mean,
        x_max=x_max,
        return_scalers=True,
    )
    start_point_normalized = (push_start_point_obj - x_mean[0]) / x_max[0]

    start_normalized = torch.tensor(start_normalized, dtype=torch.float32, device=device)
    goal_normalized = torch.tensor(goal_normalized, dtype=torch.float32, device=device)
    start_point_normalized = torch.tensor(
        start_point_normalized, dtype=torch.float32, device=device
    ).view(1, 3)
    x_mean = torch.tensor(x_mean, dtype=torch.float32, device=device)
    x_max = torch.tensor(x_max, dtype=torch.float32, device=device)

    return start_normalized, goal_normalized, start_point_normalized, x_mean, x_max


def encode_inputs(model, start_pc_normalized, goal_pc_normalized, basis, device):
    """Produce whatever start/goal representation model.forward() expects: BPS-encoded
    vectors for encoder_type="bps", or the raw normalized point clouds unchanged for
    encoder_type="ptv3" (PushVIB3S_start_point_conditioned reshapes those internally)."""
    if model.encoder_type != ENCODER_BPS:
        return start_pc_normalized, goal_pc_normalized

    if model.use_directional_bps:
        start_bps_dist, start_bps_dir = encode_pcd_with_bps(
            start_pc_normalized, basis, device=device, return_directions=True
        )
        goal_bps_dist, goal_bps_dir = encode_pcd_with_bps(
            goal_pc_normalized, basis, device=device, return_directions=True
        )
        start_bps = torch.cat(
            [start_bps_dist, start_bps_dir.reshape(start_bps_dir.shape[0], -1)], dim=1
        )
        goal_bps = torch.cat(
            [goal_bps_dist, goal_bps_dir.reshape(goal_bps_dir.shape[0], -1)], dim=1
        )
    else:
        start_bps = encode_pcd_with_bps(start_pc_normalized, basis, device=device)
        goal_bps = encode_pcd_with_bps(goal_pc_normalized, basis, device=device)

    return start_bps, goal_bps


def load_basis_tensor(basis_path, device):
    return torch.tensor(
        np.load(basis_path, allow_pickle=True),
        dtype=torch.float32,
        device=device,
    )


class DeployPushVIB3S:
    """Deployer for PushVIB3S_start_point_conditioned (encoder_type "bps" or "ptv3")."""

    def __init__(
        self,
        weights_path,
        basis_path="bps/bps_basis_r1.0_n128.npy",
        device=None,
        n_neurons=512,
        in_bps=128,
        goalD=7,
        feature_dim=128,
        use_directional_bps=True,
        num_points_per_pc=512,
        deterministic_farthest_point_sampling=True,
        strategy=STRATEGY_END_POINT,
        encoder_type=ENCODER_BPS,
        grid_size=0.01,        # PTv3 voxelization grid size (encoder_type="ptv3" only)
        order="z",             # PTv3 serialization curve order (encoder_type="ptv3" only)
        shuffle_orders=False,  # whether PTv3 shuffles between serialization orders (encoder_type="ptv3" only)
        enable_flash=False,    # whether PTv3 attention uses FlashAttention (encoder_type="ptv3" only)
    ):
        self.weights_path = Path(weights_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_type = encoder_type
        self.use_directional_bps = use_directional_bps
        self.num_points_per_pc = num_points_per_pc
        self.deterministic_farthest_point_sampling = deterministic_farthest_point_sampling

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")

        self.basis = None
        if self.encoder_type == ENCODER_BPS:
            self.basis_path = Path(basis_path)
            if not self.basis_path.exists():
                raise FileNotFoundError(f"BPS basis file not found: {self.basis_path}")
            self.basis = load_basis_tensor(self.basis_path, self.device)

        self.model = PushVIB3S_start_point_conditioned(
            n_neurons=n_neurons,
            in_bps=in_bps,
            goalD=goalD,
            feature_dim=feature_dim,
            use_directional_bps=use_directional_bps,
            deterministic=True,
            strategy=strategy,
            encoder_type=encoder_type,
            grid_size=grid_size,
            order=order,
            shuffle_orders=shuffle_orders,
            enable_flash=enable_flash,
        ).to(self.device)
        load_weights(self.model, str(self.weights_path), self.device)
        self.model.eval()

    def predict_push(
        self,
        start_point_cloud,
        goal_point_cloud,
        push_start_point,
        visualize_single=False,
        gt_end_point=None,
        visualize_distribution=False,
        n_distribution_samples=100,
    ):
        """Predict where `push_start_point` ends up, given a start/goal point cloud pair.

        Args:
            start_point_cloud: (N, 3) world-frame point cloud of the object's current state.
            goal_point_cloud: (N, 3) world-frame point cloud of the desired object state.
            push_start_point: (3,) world-frame point on the object where the push begins.
            visualize_single: if True, open an Open3D window showing the prediction.
            gt_end_point: optional (3,) world-frame ground-truth end point, drawn alongside
                the prediction when visualize_single is True.
            visualize_distribution: if True, also visualize samples from the model's prior
                and posterior over end points (requires deterministic=False internally, so
                this temporarily flips the model into stochastic mode).
            n_distribution_samples: number of samples to draw when visualize_distribution.

        Returns:
            predicted_end_point: (3,) world-frame predicted end point.
            predicted_displacement: (3,) world-frame displacement (end - push_start_point).
        """
        start_point_cloud = to_numpy_point_cloud(start_point_cloud)
        goal_point_cloud = to_numpy_point_cloud(goal_point_cloud)
        push_start_point = to_numpy_point(push_start_point)

        (
            start_pc_obj,
            goal_pc_obj,
            push_start_point_obj,
            rotation_matrix,
            tissue_mean,
        ) = objectframeize_inputs(start_point_cloud, goal_point_cloud, push_start_point)

        start_pc_obj, goal_pc_obj = downsample_point_clouds(
            start_pc_obj,
            goal_pc_obj,
            self.num_points_per_pc,
            self.deterministic_farthest_point_sampling,
        )

        (
            start_pc_normalized,
            goal_pc_normalized,
            start_point_normalized,
            x_mean,
            x_max,
        ) = normalize_inputs(start_pc_obj, goal_pc_obj, push_start_point_obj, self.device)

        start_input, goal_input = encode_inputs(
            self.model, start_pc_normalized, goal_pc_normalized, self.basis, self.device
        )

        with torch.no_grad():
            outputs = self.model(
                start_input, goal_input, start_point_normalized, x_mean=x_mean, x_max=x_max
            )
            predicted_end_point_obj = outputs["end_point"].squeeze(0).detach().cpu().numpy()

        predicted_end_point = deobjectframeize_point(
            predicted_end_point_obj, rotation_matrix, tissue_mean
        )
        predicted_displacement = predicted_end_point - push_start_point

        if visualize_distribution:
            was_deterministic = self.model.deterministic
            self.model.deterministic = False
            with torch.no_grad():
                prior_end_points_obj = self.model.sample_from_prior(
                    start_input,
                    start_point_normalized,
                    x_mean=x_mean,
                    x_max=x_max,
                    num_samples=n_distribution_samples,
                ).squeeze(0).cpu().numpy()
                posterior_end_points_obj = self.model.sample_from_posterior(
                    start_input,
                    goal_input,
                    start_point=start_point_normalized,
                    x_mean=x_mean,
                    x_max=x_max,
                    num_samples=n_distribution_samples,
                ).squeeze(0).cpu().numpy()
            self.model.deterministic = was_deterministic

            prior_end_points = [
                deobjectframeize_point(p, rotation_matrix, tissue_mean) for p in prior_end_points_obj
            ]
            posterior_end_points = [
                deobjectframeize_point(p, rotation_matrix, tissue_mean)
                for p in posterior_end_points_obj
            ]

            visualize_action_prediction(
                start_point=push_start_point,
                end_points=prior_end_points,
                start_pc=start_point_cloud,
                goal_pc=None,
                gt_end_point=None,
            )
            visualize_action_prediction(
                start_point=push_start_point,
                end_points=posterior_end_points,
                start_pc=start_point_cloud,
                goal_pc=goal_point_cloud,
                gt_end_point=gt_end_point,
            )

        if visualize_single:
            visualize_action_prediction(
                start_point=push_start_point,
                end_points=[predicted_end_point],
                start_pc=start_point_cloud,
                goal_pc=goal_point_cloud,
                gt_end_point=gt_end_point,
            )

        return predicted_end_point, predicted_displacement


class DeployPushHeuristic:
    """No-model baseline: pushes from `push_start_point` straight toward the goal centroid,
    starting from just inside the object's convex hull along that direction."""

    def predict_push(
        self, start_point_cloud, goal_point_cloud, push_start_point, visualize_single=False
    ):
        start_point_cloud = to_numpy_point_cloud(start_point_cloud)
        goal_point_cloud = to_numpy_point_cloud(goal_point_cloud)
        push_start_point = to_numpy_point(push_start_point)

        goal_mean = np.mean(goal_point_cloud, axis=0)
        adjusted_start_point = compute_hull_exit_start_point(
            start_point_cloud,
            push_start_point,
            push_start_point - goal_mean,
        ).astype(np.float32)
        predicted_end_point = goal_mean.astype(np.float32)
        predicted_displacement = predicted_end_point - adjusted_start_point

        if visualize_single:
            visualize_action_prediction(
                start_point=adjusted_start_point,
                end_points=[predicted_end_point],
                start_pc=start_point_cloud,
                goal_pc=goal_point_cloud,
            )

        return predicted_end_point, predicted_displacement


if __name__ == "__main__":

    # ENCODER_BPS or ENCODER_PTV3 -- must match how weights_path below was trained.
    encoder_type = ENCODER_BPS

    weights_path = (
        "checkpoints/realsense_bg0.0005_weight1.5_w_no_diff_dataend_point_bps_local_"
        "20260827_131126/checkpoint_429_0.020384.pt"
    )
    basis_path = "bps/bps_basis_r1.0_n128.npy"

    test_dir = Path("/home/britton/PushVIBES/data/all_good_realsense_data/test")

    visualize_single = True
    visualize_distribution = False

    episode_paths = find_pickle_files(str(test_dir))
    if not episode_paths:
        raise FileNotFoundError(f"No pickle files found in {test_dir}")

    deployer = DeployPushVIB3S(
        weights_path=weights_path,
        basis_path=basis_path,
        encoder_type=encoder_type,
    )

    for episode_path in episode_paths:
        with open(episode_path, "rb") as f:
            episode = pickle.load(f)

        # start_position: the push origin, in the same world frame as start_pointcloud.
        # displacement: ground-truth push displacement, used here only for visualization.
        push_start_point = episode["start_position"]
        gt_end_point = None
        if "displacement" in episode:
            gt_end_point = np.asarray(push_start_point).reshape(3) + np.asarray(
                episode["displacement"]
            ).reshape(3)

        predicted_end_point, predicted_displacement = deployer.predict_push(
            episode["start_pointcloud"],
            episode["goal_pointcloud"],
            push_start_point,
            visualize_single=visualize_single,
            gt_end_point=gt_end_point,
            visualize_distribution=visualize_distribution,
        )

        print(f"{episode_path}: predicted_end_point={predicted_end_point}")
