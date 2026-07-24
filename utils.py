#!/usr/bin/env python

from sklearn.decomposition import PCA
import numpy as np


import numpy as np
from sklearn.decomposition import PCA
import open3d as o3d
import pickle
import random
import torch
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def is_homogeneous_matrix(matrix):
    # Check matrix shape
    if matrix.shape != (4, 4):
        return False

    # Check last row
    if not np.allclose(matrix[3, :], [0, 0, 0, 1]):
        return False

    # Check rotational part (3x3 upper-left submatrix)
    rotational_matrix = matrix[:3, :3]
    if not np.allclose(np.dot(rotational_matrix, rotational_matrix.T), np.eye(3), atol=1.e-6) or \
            not np.isclose(np.linalg.det(rotational_matrix), 1.0, atol=1.e-6):
        
        print(np.linalg.inv(rotational_matrix), "\n")
        print(rotational_matrix.T)        
        print(np.linalg.det(rotational_matrix))
        
        return False

    return True


def find_pca_axes(obj_cloud, verbose=False):
    '''
    Given a point cloud determine a valid, right-handed coordinate frame
    '''
    pca_operator = PCA(n_components=3, svd_solver='full')
    pca_operator.fit(obj_cloud)
    centroid = np.matrix(pca_operator.mean_).T
    x_axis = pca_operator.components_[0]
    y_axis = pca_operator.components_[1]
    z_axis = np.cross(x_axis,y_axis)

    if verbose:
        print('PCA centroid', centroid)
        print('x_axis', x_axis)
        print('y_axis', y_axis)
        print('z_axis', z_axis)
    return np.array([x_axis, y_axis, z_axis]), centroid


#Compute angles between two vectors, code is from:
#https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotation_matrix_from_vectors(vec1, vec2):
    """Return rotation matrix that aligns vec1 to vec2."""
    a = vec1 / (np.linalg.norm(vec1) + 1e-12)
    b = vec2 / (np.linalg.norm(vec2) + 1e-12)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -0.999999:
        orthogonal = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v = np.cross(a, orthogonal)
        v = v / (np.linalg.norm(v) + 1e-12)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(v * np.pi)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))


def find_min_ang_vec(world_vec, cam_vecs):
    min_ang = float('inf')
    min_ang_idx = -1
    min_ang_vec = None
    for i in range(cam_vecs.shape[1]):
        angle = angle_between(world_vec, cam_vecs[:, i])
        larger_half_pi = False
        if angle > np.pi * 0.5:
            angle = np.pi - angle
            larger_half_pi = True
        if angle < min_ang:
            min_ang = angle
            min_ang_idx = i
            if larger_half_pi:
                min_ang_vec = -cam_vecs[:, i]
            else:
                min_ang_vec = cam_vecs[:, i]

    return min_ang_vec, min_ang_idx


def compute_world_to_object_frame_transformation(obj_cloud, verbose=True):
    '''
    For the given object cloud, build an object frame using PCA and aligning to the
    world frame.
    Returns a transformation from world frame to object frame.
    '''

    # Use PCA to find a starting object frame/centroid.
    axes, centroid = find_pca_axes(obj_cloud, verbose)
    axes = np.matrix(np.column_stack(axes))

    # Rotation from object frame to frame.
    R_o_w = np.eye(3)
    
    # x axes.
    x_axis = axes[:, 0]
    R_o_w[0, 0] = x_axis[0, 0]
    R_o_w[1, 0] = x_axis[1, 0]
    R_o_w[2, 0] = x_axis[2, 0]

    # y axes
    y_axis = axes[:, 1]
    R_o_w[0, 1] = y_axis[0, 0]
    R_o_w[1, 1] = y_axis[1, 0]
    R_o_w[2, 1] = y_axis[2, 0]

    # z axes
    z_axis = axes[:, 2]
    R_o_w[0, 2] = z_axis[0, 0]
    R_o_w[1, 2] = z_axis[1, 0]
    R_o_w[2, 2] = z_axis[2, 0]

    # Transpose to get rotation from world to object frame.
    R_w_o = np.transpose(R_o_w)
    d_w_o_o = np.dot(-R_w_o, centroid)
    
    # Build full transformation matrix.
    trans_matrix = np.eye(4)
    trans_matrix[:3,:3] = R_w_o
    trans_matrix[0,3] = d_w_o_o[0]
    trans_matrix[1,3] = d_w_o_o[1]
    trans_matrix[2,3] = d_w_o_o[2]    

    return trans_matrix





def rotate_around_z(ht_matrix, angle):
    rotation_z = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle),  np.cos(angle), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])
    return np.dot(rotation_z, ht_matrix)


def invert_transformation_matrix(transformation_matrix):
    # Invert the rotation part by transposing the 3x3 top-left submatrix
    R_inv = transformation_matrix[:3, :3].T
    t = transformation_matrix[:3, 3]
    
    # Invert the translation part
    t_inv = -R_inv.dot(t)
    
    # Construct the inverted transformation matrix
    inverted_matrix = np.eye(4)
    inverted_matrix[:3, :3] = R_inv
    inverted_matrix[:3, 3] = t_inv
    
    return inverted_matrix


def transform_deformernet_action(action_translation_object, T_world_to_object):
    R_world_to_object = T_world_to_object[:3, :3]
    R_object_to_world = R_world_to_object.T # inverse of rotation matrix is its transpose
    assert action_translation_object.shape == (3,)

    action_translation_world = np.dot(R_object_to_world, action_translation_object)
  
    return action_translation_world  # shape (3,)


# def transform_deformernet_action(action_translation, action_rotation, T_world_to_object):
#     """
#     Inputs:
#     action_translation.shape: (3,)
#     action_rotation.shape: (3,3)
#     tf_matrix.shape: (4,4)

#     Output: transformed_action.shape: (4,4)
#     """
#     T_object_to_eef = compose_4x4_homo_mat(action_rotation, action_translation)  # shape (4,4)
#     modified_T_world_to_object = T_world_to_object.copy()
#     modified_T_world_to_object[:3,3] = 0

#     transformed_action = compute_transformed_action(modified_T_world_to_object, T_object_to_eef)

#     assert is_homogeneous_matrix(transformed_action)
#     return transformed_action


# def compute_transformed_action(T_world_to_object, T_object_to_eef):
#     # Compute the transformation matrix
#     T_world_to_eef = np.dot(T_world_to_object, T_object_to_eef)

#     return T_world_to_eef

# def transform_point_cloud(point_cloud, transformation_matrix):
#     # Add homogeneous coordinate (4th component) of 1 to each point
#     homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

#     # Apply the transformation matrix to each point
#     transformed_points = np.dot(homogeneous_points, transformation_matrix.T)

#     # Remove the homogeneous coordinate (4th component) from the transformed points
#     transformed_points = transformed_points[:, :3]

#     return transformed_points


def transform_point_cloud(point_cloud, transformation_matrix):
    """
    Transform a point cloud using a 4x4 transformation matrix.

    Parameters:
    - point_cloud (numpy.ndarray): Point cloud data with shape (N, 3).
    - transformation_matrix (numpy.ndarray): 4x4 transformation matrix.

    Returns:
    - numpy.ndarray: Transformed point cloud data with shape (N, 3).
    """
    # Add homogeneous coordinate (4th component) of 1 to each point
    homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # print("homogeneous_points.shape:", homogeneous_points.shape) # (N, 4)
    # print("transformation_matrix.shape:", transformation_matrix.shape) # (4, 4)

    # Apply the transformation matrix to each point
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)

    # Remove the homogeneous coordinate (4th component) from the transformed points
    transformed_points = transformed_points[:, :3]

    return transformed_points


def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def save_val_error_plot(sim_val_error_history, real_val_error_history, checkpoint_dir, epoch):
    real_epochs = list(range(1, len(real_val_error_history["total"]) + 1))
    sim_epochs = list(range(1, len(sim_val_error_history["total"]) + 1))

    plt.figure(figsize=(10, 6))

    sim_color = "tab:blue"
    real_color = "tab:orange"

    if sim_epochs:
        plt.plot(sim_epochs, sim_val_error_history["total"], color=sim_color, linestyle="-", label="sim total")
        plt.plot(sim_epochs, sim_val_error_history["end_point"], color=sim_color, linestyle="--", label="sim end point")

    plt.plot(real_epochs, real_val_error_history["total"], color=real_color, linestyle="-", label="real total")
    plt.plot(real_epochs, real_val_error_history["end_point"], color=real_color, linestyle="--", label="real end point")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Error")
    plt.title("Validation Error Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(checkpoint_dir, f"val_error_curve.png")
    plt.savefig(out_path)
    plt.close()


def save_val_loss_plot(sim_val_loss_history, real_val_loss_history, checkpoint_dir, epoch):
    epochs = list(range(1, len(real_val_loss_history["total"]) + 1))

    plt.figure(figsize=(10, 6))

    sim_color = "tab:blue"
    real_color = "tab:orange"

    sim_epochs = list(range(1, len(sim_val_loss_history["total"]) + 1))
    if sim_epochs:
        plt.plot(sim_epochs, sim_val_loss_history["total"], color=sim_color, linestyle="-", label="sim total")
        plt.plot(sim_epochs, sim_val_loss_history["end_point"], color=sim_color, linestyle="--", label="sim end point")
        plt.plot(sim_epochs, sim_val_loss_history["kl"], color=sim_color, linestyle="-.", label="sim kl (effective)")

    plt.plot(epochs, real_val_loss_history["total"], color=real_color, linestyle="-", label="real total")
    plt.plot(epochs, real_val_loss_history["end_point"], color=real_color, linestyle="--", label="real end point")
    plt.plot(epochs, real_val_loss_history["kl"], color=real_color, linestyle="-.", label="real kl (effective)")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(checkpoint_dir, f"val_loss_curve.png")
    plt.savefig(out_path)
    plt.close()


def save_weight_comparison_boxplot(results_rows, output_path):
    real_start_plot_data = []
    real_displacement_plot_data = []
    labels = []

    for row in results_rows:
        real_start_errors = np.asarray(row.get("real_start_point_errors", []), dtype=float)
        real_displacement_errors = np.asarray(row.get("real_displacement_errors", []), dtype=float)

        if real_start_errors.size == 0 or real_displacement_errors.size == 0:
            continue

        labels.append(row["model_name"])
        real_start_plot_data.append(real_start_errors * 100.0)
        real_displacement_plot_data.append(real_displacement_errors * 100.0)

    if not labels:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    boxplot_style = dict(
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"facecolor": "#7db7d8", "edgecolor": "#2f5d73"},
        whiskerprops={"color": "#2f5d73"},
        capprops={"color": "#2f5d73"},
        flierprops={"marker": "o", "markerfacecolor": "#d95f02", "markeredgecolor": "#d95f02", "markersize": 4, "alpha": 0.5},
    )

    axes[0].boxplot(real_start_plot_data, labels=labels, **boxplot_style)
    axes[0].set_title("Start Point Error for Unseen Real Data")
    axes[0].set_ylabel("Error (cm)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].boxplot(real_displacement_plot_data, labels=labels, **boxplot_style)
    axes[1].set_title("Displacement Error for Unseen Real Data")
    axes[1].set_ylabel("Error (cm)")
    axes[1].set_xlabel("Weights")
    axes[1].grid(True, axis="y", alpha=0.3)

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=800, bbox_inches="tight")
    plt.close(fig)


def objectframeize_point_clouds(start_pc, goal_pc, start_point=None, goal_point=None):
    """
    Objectframeize point clouds

    takes in np arrays and returns np arrays in an objectframeized format
    """
    # Compute the mean of the tissue point cloudpr 
    tissue_mean = np.mean(start_pc, axis=0)

    # Shift all point clouds so that the tissue mean is at the origin
    start_pc -= tissue_mean
    goal_pc -= tissue_mean 
    if start_point is not None:
        start_point -= tissue_mean
    if goal_point is not None:
        goal_point -= tissue_mean

    # Perform PCA on the start point cloud
    pca = PCA(n_components=3)
    pca.fit(start_pc)

    # Create rotation matrix from PCA components
    rotation_matrix = pca.components_.T


    start = np.dot(start_pc, rotation_matrix)
    goal = np.dot(goal_pc, rotation_matrix)
    if start_point is not None:
        start_point = np.dot(start_point, rotation_matrix)
    if goal_point is not None:
        goal_point = np.dot(goal_point, rotation_matrix)
    displacement = None
    if start_point is not None and goal_point is not None:
        displacement = goal_point - start_point

    return start, goal, start_point, goal_point, displacement


def objectframeize_start_pc(start_pc):
    """
    Objectframeize point clouds

    takes in np arrays and returns np arrays in an objectframeized format
    """
    # Compute the mean of the tissue point cloudpr 
    tissue_mean = np.mean(start_pc, axis=0)

    # Shift all point clouds so that the tissue mean is at the origin
    start_pc -= tissue_mean

    # Perform PCA on the start point cloud
    pca = PCA(n_components=3)
    pca.fit(start_pc)

    # Create rotation matrix from PCA components
    rotation_matrix = pca.components_.T

    # Flip the z-axis (camera frame opposite)


    start = np.dot(start_pc, rotation_matrix)

    return start, rotation_matrix, tissue_mean

def deobjectframeize_point(point, rotation_matrix, tissue_mean):
    """
    Transform a point from object frame back to global frame
    
    Args:
        point: numpy array of shape (3,) representing point in object frame
        rotation_matrix: numpy array of shape (3,3) from objectframeize_start_pc
        tissue_mean: numpy array of shape (3,) representing original tissue mean
        
    Returns:
        numpy array of shape (3,) representing point in global frame
    """
    # Rotate point back using transpose of rotation matrix
    point_rotated = np.dot(point, rotation_matrix.T)
    
    # Translate back using tissue mean
    point_global = point_rotated + tissue_mean
    
    return point_global

def objectframeize_goal_pc(goal_pc, rotation_matrix, tissue_mean):
    """
    Transform goal point cloud into object frame using same parameters as start point cloud
    
    Args:
        goal_pc: numpy array of shape (N,3) representing goal point cloud
        rotation_matrix: numpy array of shape (3,3) from objectframeize_start_pc
        tissue_mean: numpy array of shape (3,) representing tissue mean from start pc
        
    Returns:
        numpy array of shape (N,3) representing goal point cloud in object frame
    """
    # Shift using same tissue mean as start point cloud
    goal_pc -= tissue_mean
    
    # Rotate using same rotation matrix as start point cloud
    goal = np.dot(goal_pc, rotation_matrix)
    
    return goal





def visualize_action_prediction(start_point, end_points, start_pc, goal_pc, gt_end_point=None, visualize_origin=False, object_frame_visualization=True):
    """
    Visualize a push/grasp prediction: start and goal point clouds, a line from start to each
    predicted end point, and a cone arrowhead at each end point tip.

    Args:
        start_point (np.ndarray): (3,) push start position in world coordinates
        end_points: (3,) array or list of (3,) arrays — predicted end positions
        start_pc (np.ndarray): (N, 3) start point cloud
        goal_pc (np.ndarray): (N, 3) goal point cloud
        gt_end_point (np.ndarray, optional): (3,) ground truth end position
    """
    if isinstance(end_points, np.ndarray) and end_points.ndim == 1:
        end_points = [end_points]

    start_point = np.asarray(start_point).reshape(3)

    start_pcd = o3d.geometry.PointCloud()
    start_pcd.points = o3d.utility.Vector3dVector(start_pc)
    start_pcd.paint_uniform_color([1, 0, 0])

    goal_pcd = o3d.geometry.PointCloud()
    goal_pcd.points = o3d.utility.Vector3dVector(goal_pc)
    goal_pcd.paint_uniform_color([0, 0, 1])

    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
    start_sphere.translate(start_point)
    start_sphere.paint_uniform_color([1, 0, 0])

    geometries = [start_pcd, goal_pcd, start_sphere]

    cone_radius = 0.001
    cone_height = 0.003

    for ep in end_points:
        ep = np.asarray(ep).reshape(3)
        direction = ep - start_point

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array([start_point, ep]))
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_set.paint_uniform_color([0.5, 1.0, 0.0])
        geometries.append(line_set)

        cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
        if np.linalg.norm(direction) > 1e-9:
            rot = rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), direction)
            cone.rotate(rot, center=(0.0, 0.0, 0.0))
            direction_unit = direction / np.linalg.norm(direction)
            cone.translate(ep - direction_unit * cone_height)
        else:
            cone.translate(ep)
        cone.paint_uniform_color([0, 0, 1])
        geometries.append(cone)

    if gt_end_point is not None:
        gt_end_point = np.asarray(gt_end_point).reshape(3)
        gt_direction = gt_end_point - start_point

        gt_line = o3d.geometry.LineSet()
        gt_line.points = o3d.utility.Vector3dVector(np.array([start_point, gt_end_point]))
        gt_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        gt_line.paint_uniform_color([0.7, 0.85, 1.0])
        geometries.append(gt_line)

        gt_cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
        if np.linalg.norm(gt_direction) > 1e-9:
            rot = rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), gt_direction)
            gt_cone.rotate(rot, center=(0.0, 0.0, 0.0))
            gt_direction_unit = gt_direction / np.linalg.norm(gt_direction)
            gt_cone.translate(gt_end_point - gt_direction_unit * cone_height)
        else:
            gt_cone.translate(gt_end_point)
        gt_cone.paint_uniform_color([0.7, 0.85, 1.0])
        geometries.append(gt_cone)

    if visualize_origin:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

    if object_frame_visualization:
        o3d.visualization.draw_geometries(
            geometries,
            zoom=0.9,
            front=[-0.508, -0.858, -0.077],
            lookat=[0.015, 0.015, 0.002],
            up=[-0.031, -0.071, 0.997],
        )
    else:
        o3d.visualization.draw_geometries(
            geometries
        )


def down_sampling(pc, num_pts=1024, return_indices=False):


   """
   Input:
       pc: point cloud data, [B, N, D] where B = num batches, N = num points, D = feature size (typically D=3)
       num_pts: number of samples
   Return:
       centroids (numpy.ndarray): sampled pointcloud index, [num_pts, D]
       pc (numpy.ndarray): down_sampled point cloud, [num_pts, D]
   """


   if pc.ndim == 2:
       # insert batch_size axis
       pc = np.copy(pc)[None, ...]


   B, N, D = pc.shape
   xyz = pc[:, :,:3]
   centroids = np.zeros((B, num_pts))
   distance = np.ones((B, N)) * 1e10
   farthest = np.random.uniform(low=0, high=N, size=(B,)).astype(np.int32)


   for i in range(num_pts):
       centroids[:, i] = farthest
       centroid = xyz[np.arange(0, B), farthest, :] # (B, D)
       centroid = np.expand_dims(centroid, axis=1) # (B, 1, D)
       dist = np.sum((xyz - centroid) ** 2, -1) # (B, N)
       mask = dist < distance
       distance[mask] = dist[mask]
       farthest = np.argmax(distance, -1) # (B,)


   pc = pc[np.arange(0, B).reshape(-1, 1), centroids.astype(np.int32), :]


   if return_indices:
       return pc.squeeze(), centroids.astype(np.int32)


   return pc.squeeze()


def farthest_point_sampling(pc, num_pts=1024, return_indices=False, use_internally_random_seed=False):
    """
    Farthest Point Sampling (FPS) downsampling for a SINGLE point cloud (no batch dim),
    with:
      - exact XYZ deduplication
      - duplicate count reporting
      - optional internal randomness override

    If use_internally_random_seed=True:
        The function will IGNORE any externally-set random seed and behave non-deterministically
        across calls (i.e., different outputs each run).

    Uses Open3D's farthest_point_down_sample internally (~5x faster than a pure-Python
    greedy loop); index recovery back into pc's original rows is only done when needed
    (return_indices=True or extra columns beyond xyz), since it costs a bit more than the
    common xyz-only path.

    Args:
        pc: (N, D) numpy array
        num_pts: number of points to sample (M)
        return_indices: if True, also return sampled indices (in ORIGINAL pc index space)
        use_internally_random_seed: if True, override global RNG determinism

    Returns:
        sampled_pc: (M, D)
        sampled_idx (optional): (M,) indices into ORIGINAL pc
    """

    # -----------------------------
    # Internal RNG control
    # -----------------------------
    if use_internally_random_seed:
        # Create a fresh RNG seeded from OS entropy (non-deterministic)
        rng = np.random.default_rng()   # independent RNG
        rand_int = lambda low, high: rng.integers(low, high)
    else:
        # Use global numpy RNG (deterministic if np.random.seed(...) was set externally)
        rand_int = lambda low, high: np.random.randint(low, high)

    # -----------------------------
    # 1) Deduplicate exact XYZ rows
    # -----------------------------
    if pc.ndim != 2:
        raise ValueError(f"Expected pc to have shape (N, D); got shape {pc.shape}")

    N, D = pc.shape
    if D < 3:
        raise ValueError(f"Expected pc to have at least 3 dims (xyz); got D={D}")

    xyz = pc[:, :3]  # (N, 3)

    # Find unique XYZ rows and keep first occurrence
    _, unique_first_idx = np.unique(xyz, axis=0, return_index=True)
    unique_first_idx = np.sort(unique_first_idx)

    num_dups = N - unique_first_idx.shape[0]

    # Print only the count
    # if num_dups > 0:
    #     print(f"[farthest_point_sampling] Duplicate XYZ points removed: {num_dups}")

    # Deduplicated point cloud
    pc_dedup = pc[unique_first_idx]
    xyz = pc_dedup[:, :3]

    # Mapping: dedup index -> original index
    dedup_to_orig = unique_first_idx

    N_dedup = pc_dedup.shape[0]
    if N_dedup == 0:
        raise ValueError("down_sampling: point cloud is empty after deduplication.")

    # Clamp num_pts if needed
    if num_pts > N_dedup:
        print(
            f"[down_sampling] Requested num_pts={num_pts} > unique points={N_dedup}. "
            f"Clamping to {N_dedup}."
        )
        num_pts = N_dedup

    # ------------------------------------
    # 2) Farthest Point Sampling (no batch)
    # ------------------------------------
    start_index = int(rand_int(0, N_dedup))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    sampled_pcd = pcd.farthest_point_down_sample(num_pts, start_index)
    sampled_xyz = np.asarray(sampled_pcd.points, dtype=xyz.dtype)

    if D == 3 and not return_indices:
        # Fast path: nothing beyond xyz to preserve, and indices weren't requested.
        return sampled_xyz.astype(pc.dtype)

    # Recover indices into pc_dedup. FPS only ever selects existing points (no
    # interpolation), so exact xyz matching reliably maps samples back to their source row.
    lookup = {tuple(row): i for i, row in enumerate(xyz)}
    centroids = np.array([lookup[tuple(row)] for row in sampled_xyz], dtype=np.int64)

    # Gather sampled points (preserves any columns beyond xyz)
    sampled_pc = pc_dedup[centroids]  # (M, D)

    # Optionally return original indices
    if return_indices:
        sampled_idx_orig = dedup_to_orig[centroids]
        return sampled_pc, sampled_idx_orig

    return sampled_pc



def objectframeize_point_cloud_difDef(tissue, context, flip_x=False):
    # Compute the mean of the tissue point cloud
    tissue_mean = np.mean(tissue, axis=0)
    
    tissue -= tissue_mean
    context -= tissue_mean

    # Perform PCA on the context point cloud
    pca = PCA(n_components=3)
    pca.fit(context)

    # Create rotation matrix from PCA components
    rotation_matrix = pca.components_.T

    # # Flip the z-axis (camera frame opposite)
    rotation_matrix[:, 2] = -rotation_matrix[:, 2]
    
    if flip_x:
        rotation_matrix[:, 0] = -rotation_matrix[:, 0]

    tissue = np.dot(tissue, rotation_matrix)
    context = np.dot(context, rotation_matrix)


        

    # this is to ensure that the goal is always on the same side of the tissue (down the trachea)

    # Scale all point clouds to unit size
    # Get max distance from origin across all point clouds
    max_dist = 0
    for points in [tissue, context]:
        if len(points) > 0:
            dist = np.max(np.linalg.norm(points, axis=1))
            max_dist = max(max_dist, dist)
    


    # # Scale all points by reciprocal of max distance
    # scale_factor = 1.0 / max_dist
    # tissue *= scale_factor
    # context *= scale_factor 

    return tissue, context, rotation_matrix, tissue_mean


def goal_to_camera_frame(goal, rotation_matrix, tissue_mean):
    # Rotate back using the transpose of the rotation matrix
    goal = np.dot(goal, rotation_matrix.T)

    # Shift back using the tissue mean
    goal = goal + tissue_mean

    return goal





def scale_point_clouds(pc1, pc2):
    """
    Scale point clouds to unit size and return scale factor.
    
    Args:
        pc1 (np.ndarray): First point cloud
        pc2 (np.ndarray): Second point cloud
        
    Returns:
        pc1_norm (np.ndarray): Normalized first point cloud
        pc2_norm (np.ndarray): Normalized second point cloud 
        scale_factor (float): Scale factor used for normalization
    """
    # # Get max distance from origin across both point clouds
    # max_dist = 0
    # for points in [pc1, pc2]:
    #     if len(points) > 0:
    #         dist = np.max(np.linalg.norm(points, axis=1))
    #         max_dist = max(max_dist, dist)

    # # Scale point clouds by reciprocal of max distance
    # scale_factor = 1.0 / max_dist
    # pc1_norm = pc1 * scale_factor
    # pc2_norm = pc2 * scale_factor

    # Scale point clouds by 1/1000
    scale_factor = 1/1000
    pc1_norm = pc1 * scale_factor
    pc2_norm = pc2 * scale_factor

    return pc1_norm, pc2_norm, scale_factor


def normalize_point_clouds(pc1, pc2):
    """
    Scale point clouds to unit size and return scale factor.
    
    Args:
        pc1 (np.ndarray): First point cloud
        pc2 (np.ndarray): Second point cloud
        
    Returns:
        pc1_norm (np.ndarray): Normalized first point cloud
        pc2_norm (np.ndarray): Normalized second point cloud 
        scale_factor (float): Scale factor used for normalization
    """
    # Get max distance from origin across both point clouds
    max_dist = 0
    for points in [pc1, pc2]:
        if len(points) > 0:
            dist = np.max(np.linalg.norm(points, axis=1))
            max_dist = max(max_dist, dist)

    # Scale point clouds by reciprocal of max distance
    scale_factor = 1.0 / max_dist
    pc1_norm = pc1 * scale_factor
    pc2_norm = pc2 * scale_factor


    return pc1_norm, pc2_norm, scale_factor



def filter_tumor_pc(tumor_pc):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tumor_pc)
    
    num_neighbors = 10
    std_ratio = 2.0 # points beyond this std are outliers
    pcd, _ = pcd.remove_statistical_outlier(num_neighbors, std_ratio)

    eps = 0.0005 # max distance to be in the same cluster
    min_points = 20 # min number of points in a cluster
    labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=True))
    valid_points_mask = labels != -1
    tumor_pc = pcd.select_by_index(np.where(valid_points_mask)[0])
    tumor_pc = np.asarray(tumor_pc.points)

    # radii = [0.3, 0.6, 1.2]
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radii[0], max_nn=30))
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radii[0], radii[1], radii[2]]))
    return tumor_pc




def find_pickle_files(data_dir):
    """
    Recursively find all .pickle and .pkl files under a directory.

    Args:
        data_dir (str): Root directory to search

    Returns:
        list[str]: List of absolute file paths to pickle files
    """
    file_list = []

    for root, dirs, files in os.walk(data_dir):  # recursive traversal
        for file in files:
            if file.endswith(('.pickle', '.pkl')):  # tuple = multiple suffix match
                file_list.append(os.path.join(root, file))

    return file_list


def normalize(x, known_scalers=False, x_mean=None, x_max=None, max_rescale=True, return_scalers=False):
    """
    Normalize a batch of point clouds to fit inside a unit sphere (center at the mean,
    scale by the max distance from that center).

    Args:
        x (np.ndarray): [n_clouds, n_points, n_dims] input point clouds
        known_scalers (bool): if True, use the provided x_mean/x_max instead of computing them
        x_mean (np.ndarray or None): [n_clouds, n_dims] mean to subtract per cloud (required if known_scalers=True)
        x_max (np.ndarray or None): [n_clouds, 1] max norm to divide by per cloud (required if known_scalers=True)
        max_rescale (bool): if False, only center the cloud (skip dividing by x_max)
        return_scalers (bool): whether to also return the x_mean/x_max used

    Returns:
        x_norm (np.ndarray): [n_clouds, n_points, n_dims] normalized point clouds
        x_mean (np.ndarray): [n_clouds, n_dims] mean used per cloud (only if return_scalers=True)
        x_max (np.ndarray): [n_clouds, 1] max norm used per cloud (only if return_scalers=True)
    """

    def _normalize_cloud(cloud, mean=None, max_norm=None, max_rescale=True):
        if mean is None:
            mean = np.mean(cloud, axis=0)

        cloud_norm = np.copy(cloud - mean)

        if max_norm is None:
            max_norm = np.max(np.sqrt(np.sum(np.square(cloud), axis=1))) if max_rescale else 1.0
        cloud_norm = cloud_norm / max_norm

        return cloud_norm, mean, max_norm

    n_clouds, n_points, n_dims = x.shape
    x_norm = np.zeros([n_clouds, n_points, n_dims])

    if not known_scalers:
        x_mean = np.zeros([n_clouds, n_dims])
        x_max = np.zeros([n_clouds, 1])

    for pid in range(n_clouds):
        if not known_scalers:
            x_norm[pid], x_mean[pid], x_max[pid] = _normalize_cloud(x[pid], max_rescale=max_rescale)
        else:
            x_norm[pid], _, _ = _normalize_cloud(x[pid], x_mean[pid], x_max[pid], max_rescale=max_rescale)

    if return_scalers:
        return x_norm, x_mean, x_max
    return x_norm


def denormalize_torch(x_norm, x_mean, x_max):
    """
    Inverse of normalize(), using torch tensors so it stays differentiable in the model's forward pass.

    Args:
        x_norm (torch.Tensor): [batch, ..., n_dims] normalized values (e.g. predicted points)
        x_mean (torch.Tensor): [batch, n_dims] mean that was subtracted during normalization
        x_max (torch.Tensor): [batch, 1] max norm that was divided out during normalization

    Returns:
        torch.Tensor: [batch, ..., n_dims] denormalized values
    """
    return x_norm * x_max + x_mean
