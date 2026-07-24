import os
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

def normalize(x, known_scalers=False, x_mean=None, x_max=None, max_rescale=True, return_scalers=False, verbose=False):
    """Normalize point clouds to fit a unit sphere

    Parameters
    ----------
    x : [n_clouds, n_points, n_dims]
        Input point clouds
    known_scalers: bool
        if True, use provided x_mean and x_max scalers for normalization
    max_rescale: bool
        if False, normalization will only include shifting by the mean value
    x_mean : numpy array [n_clouds, n_dims]
        if provided, mean value for every cloud used for normalization
    x_max : None or numpy array [n_clouds, 1]
        if provided, max norm for every cloud used for normalization
    return_scalers: bool
        whether to return point cloud scalers (needed for denormalisation)
    verbose: bool
        whether to print progress
    Returns
    -------
    x_norm : numpy array [n_clouds, n_points, n_dims]
        Normalized point clouds
    x_mean : numpy array [n_clouds, n_dims]
        Mean value of every cloud
    x_max : numpy array [n_clouds, 1]
        Max norm of every cloud
    """

    def _normalize_cloud(x, x_mean=None, x_max=None, max_rescale=True):
        """normalize single cloud"""

        if x_mean is None:
            x_mean = np.mean(x, axis=0)

        x_norm = np.copy(x - x_mean)

        # note: max norm could be not robust to outliers!
        if x_max is None:
            if max_rescale:
                x_max = np.max(np.sqrt(np.sum(np.square(x), axis=1)))
            else:
                x_max = 1.0
        x_norm = x_norm / x_max

        return x_norm, x_mean, x_max

    n_clouds, n_points, n_dims = x.shape

    x_norm = np.zeros([n_clouds, n_points, n_dims])

    if known_scalers is False:
        x_mean = np.zeros([n_clouds, n_dims])
        x_max = np.zeros([n_clouds, 1])

    fid_lst = range(0, n_clouds)

    if verbose:
        fid_lst = tqdm(fid_lst)

    for pid in fid_lst:
        if known_scalers is False:
            x_norm[pid], x_mean[pid], x_max[pid] = _normalize_cloud(x[pid])
        else:
            x_norm[pid], _, _ = _normalize_cloud(x[pid], x_mean[pid], x_max[pid], max_rescale=max_rescale)

    if return_scalers:
        return x_norm, x_mean, x_max
    else:
        return x_norm


def denormalize(x_norm, x_mean, x_max):
    """Denormalize point clouds

    Parameters
    ----------
    x : [n_clouds, n_points, n_dims]
        Input point clouds
    rescale: bool
        if False, normalization will only include shifting by the mean value
    x_mean : numpy array [n_clouds, n_dims]
        if provided, mean value for every cloud used for normalization
    x_max : None or numpy array [n_clouds, 1]
        if provided, max norm for every cloud used for normalization

    Returns
    -------
    x_norm : numpy array [n_clouds, n_points, n_dims]
        Normalized point clouds
    x_mean : numpy array [n_clouds, n_dims]
        Mean value of every cloud
    x_max : numpy array [n_clouds, 1]
        Max norm of every cloud
    """

    def _denormalize_cloud(x_norm, x_mean, x_max):
        """denormalize single cloud"""

        x_denorm = x_norm * x_max + x_mean

        return x_denorm

    x_denorm = np.zeros(x_norm.shape)

    for pid in range(0, len(x_norm)):
        x_denorm[pid] = _denormalize_cloud(x_norm[pid], x_mean[pid], x_max[pid])

    return x_denorm

def denormalize_torch(x_norm, x_mean, x_max):
    """Denormalize point clouds using PyTorch

    Parameters
    ----------
    x_norm : torch.Tensor [n_clouds, n_points, n_dims]
        Input normalized point clouds
    x_mean : torch.Tensor [n_clouds, n_dims]
        Mean value for every cloud used for normalization
    x_max : torch.Tensor [n_clouds, 1]
        Max norm for every cloud used for normalization

    Returns
    -------
    x_denorm : torch.Tensor [n_clouds, n_points, n_dims]
        Denormalized point clouds
    """


    return x_norm * x_max + x_mean


# was 1024 and was 1.0
def get_random_basis(n_points=512, n_dims=3, radius=0.5, random_seed=33):
    """Sample uniformly from d-dimensional unit ball

    The code is inspired by this small note:
    https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html

    Parameters
    ----------
    n_points : int
        number of samples
    n_dims : int
        number of dimensions
    radius: float
        ball radius
    random_seed: int
        random seed for basis point selection
    Returns
    -------
    x : numpy array
        points sampled from d-ball
    """

    np.random.seed(random_seed)
    # sample point from d-sphere
    x = np.random.normal(size=[n_points, n_dims])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms

    # now sample radiuses uniformly
    r = np.random.uniform(size=[n_points, 1])
    u = np.power(r, 1.0 / n_dims)
    x = radius * x_unit * u
    np.random.seed(None)

    return x.astype(np.float32)

def compute_knn(query_points, reference_points, k=1):
    """
    Custom KNN implementation using PyTorch.
    
    Args:
        query_points (torch.Tensor): Points to find neighbors for (N, P1, D)
        reference_points (torch.Tensor): Points to search in (N, P2, D)
        k (int): Number of nearest neighbors to find
    
    Returns:
        distances (torch.Tensor): Distances to k-nearest neighbors (N, P1, K)
        indices (torch.Tensor): Indices of k-nearest neighbors (N, P1, K)
        directions (torch.Tensor): Direction vectors to k-nearest neighbors (N, P1, K, D)
    """
    ref_sq = torch.sum(reference_points ** 2, dim=-1, keepdim=True)  
    query_sq = torch.sum(query_points ** 2, dim=-1, keepdim=True)    
    
    cross_term = torch.matmul(query_points, reference_points.transpose(-2, -1))
    
    distances = query_sq.transpose(-2, -1) + ref_sq - 2 * cross_term
    distances = torch.clamp(distances, min=0)  # Ensure non-negative values

    distances, indices = torch.topk(distances, k=k, dim=-1, largest=False)
    
    # Get the batch size, number of query points, and dimensionality
    batch_size, num_queries, _ = query_points.shape
    _, _, num_neighbors = indices.shape
    dim = query_points.shape[-1]
    
    # Initialize directions tensor
    directions = torch.zeros((batch_size, num_queries, num_neighbors, dim), device=query_points.device)
    
    # Create batch indices for advanced indexing
    batch_indices = torch.arange(batch_size, device=query_points.device).view(-1, 1, 1)
    batch_indices = batch_indices.expand(-1, num_queries, num_neighbors)
    
    # Create query indices for advanced indexing
    query_indices = torch.arange(num_queries, device=query_points.device).view(1, -1, 1)
    query_indices = query_indices.expand(batch_size, -1, num_neighbors)
    
    # Get the reference points corresponding to the nearest neighbors
    selected_points = reference_points[batch_indices, indices]
    
    # Calculate direction vectors: reference_point - query_point
    # This gives the direction from query point to reference point
    directions = selected_points - query_points.unsqueeze(2)
    
    return torch.sqrt(distances), indices, directions

def compute_knn_pytorch(query_points, reference_points, k=1):
    """
    Efficient KNN using PyTorch's built-in functions.

    Args:
        query_points (torch.Tensor): Points to find neighbors for (N, P1, D)
        reference_points (torch.Tensor): Points to search in (N, P2, D)
        k (int): Number of nearest neighbors to find

    Returns:
        distances (torch.Tensor): Distances to k-nearest neighbors (N, P1, K)
        indices (torch.Tensor): Indices of k-nearest neighbors (N, P1, K)
        directions (torch.Tensor): Direction vectors to k-nearest neighbors (N, P1, K, D)
    """
    # Compute pairwise distances using torch.cdist
    distances = torch.cdist(query_points, reference_points, p=2)  # Euclidean distance

    # Find top-k nearest neighbors
    distances, indices = torch.topk(distances, k=k, dim=-1, largest=False)
    
    # Calculate direction vectors
    nearest_points = torch.gather(
        reference_points.unsqueeze(1).expand(-1, query_points.size(1), -1, -1),
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, query_points.size(-1))
    )
    directions = nearest_points - query_points.unsqueeze(-2)

    return distances, indices, directions

  
def compute_knn_pytorch(query_points, reference_points, k=1):
    """
    Efficient KNN using PyTorch's built-in functions.

    Args:
        query_points (torch.Tensor): Points to find neighbors for (N, P1, D)
        reference_points (torch.Tensor): Points to search in (N, P2, D)
        k (int): Number of nearest neighbors to find

    Returns:
        distances (torch.Tensor): Distances to k-nearest neighbors (N, P1, K)
        indices (torch.Tensor): Indices of k-nearest neighbors (N, P1, K)
        directions (torch.Tensor): Direction vectors to k-nearest neighbors (N, P1, K, D)
    """
    # Compute pairwise distances using torch.cdist
    distances = torch.cdist(query_points, reference_points, p=2)  # Euclidean distance

    # Find top-k nearest neighbors
    distances, indices = torch.topk(distances, k=k, dim=-1, largest=False)
    
    # Calculate direction vectors
    nearest_points = torch.gather(
        reference_points.unsqueeze(1).expand(-1, query_points.size(1), -1, -1),
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, query_points.size(-1))
    )
    directions = nearest_points - query_points.unsqueeze(-2)

    return distances, indices, directions


def encode_pcd_with_bps(x, bps, device='cpu', return_directions=False):
    """
    BPS encoding using custom KNN implementation.

    Args:
        x (torch.Tensor): Point cloud data of shape (N, P_x, D)
        bps (torch.Tensor): BPS points of shape (P_bps, D)
        device (str): Device to use for computation
        return_directions (bool): Whether to return direction vectors

    Returns:
        torch.Tensor or tuple: 
            - If return_directions=False: BPS encoding of shape (N, P_bps)
            - If return_directions=True: Tuple of (distances, directions) where
              distances has shape (N, P_bps) and directions has shape (N, P_bps, D)
    """
    # Ensure inputs are tensors on GPU
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(bps):
        bps = torch.tensor(bps, dtype=torch.float32)

    x = x.to(device)
    bps = bps.to(device)

    # Handle non-batch input
    if x.ndim == 2:
        x = x.unsqueeze(0)

    # Expand BPS points to match batch size
    bps_expanded = bps.unsqueeze(0).expand(x.shape[0], -1, -1)

    # Find nearest neighbors using custom KNN
    distances, _indices, scaled_directions = compute_knn_pytorch(bps_expanded, x, k=1)
    distances = distances.squeeze(-1)
    
    # If directions are not needed, return only distances (backward compatibility)
    if not return_directions:
        return distances
        
    
    # verify_knn(x, bps, scaled_directions)

    return distances, scaled_directions


def verify_knn(original_points, bps, directions):
    """
    Verify the distances and directions from the KNN computation by visualizing the original points compared to distance and direction offset from BPS points.
    """

    # Convert tensors to numpy arrays
    original_points_np = original_points.detach().cpu().numpy().squeeze()
    bps_np = bps.cpu().numpy()
    directions_np = directions.detach().cpu().numpy().squeeze()

    points_relative_to_bps = bps_np + directions_np #* np.tile(distances_np[:, np.newaxis], (1, 3))

    # Use open3d for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_points_np)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for original points

    # Create BPS points
    bps_relative_pcd = o3d.geometry.PointCloud()
    bps_relative_pcd.points = o3d.utility.Vector3dVector(points_relative_to_bps)
    bps_relative_pcd.paint_uniform_color([1, 0, 0])  # Red color for BPS-relative points

    # Bps
    bps_pcd = o3d.geometry.PointCloud()
    bps_pcd.points = o3d.utility.Vector3dVector(bps_np)
    bps_pcd.paint_uniform_color([0, 1, 0])  # Green color for BPS points

    # Visualize
    o3d.visualization.draw_geometries([pcd, bps_relative_pcd],
                                      window_name="KNN Verification",
                                      width=800,
                                      height=600,
                                      left=50,
                                      top=50,
                                      mesh_show_back_face=True)


def recover_pointcloud_from_bps(bps_encoded_pc_with_direction, bps):
    """
    
    Recover the original point cloud from BPS encoding with direction vectors.

    Args:
        bps_encoded_pc_with_direction (torch.Tensor): BPS encoded point cloud with direction vectors of shape (batch_size, number_of_bps_points * 4)
        1 value for distance and 3 values for direction (x, y, z), for each point
        Note: The direction vector in the current version is not normalized.
    """

    # reshape to (batch_size, number_of_bps_points, 4)
    split_index = 512
    _recovered_dist = bps_encoded_pc_with_direction[:, :, :split_index]
    recovered_dir_flat = bps_encoded_pc_with_direction[:, :, split_index:]
    recovered_dir = recovered_dir_flat.reshape(bps_encoded_pc_with_direction.shape[0], 512, 3)

    recovered_pc = bps + recovered_dir[0]  # Add direction vectors to BPS points

    return recovered_pc

if __name__ == "__main__":
    # Example usage
    radius = 1.0
    num_points = 128  # Number of basis points
    bps = get_random_basis(n_points=num_points, n_dims=3, radius=radius)
    basis_dir = "./bps"
    basis_path = os.path.join(basis_dir, f"bps_basis_r{radius}_n{num_points}.npy")

    os.makedirs(os.path.dirname(basis_dir), exist_ok=True)
    print(f"Created and saved new BPS basis at {basis_path}")
    print(f"Basis shape: {bps.shape}")
    np.save(basis_path, bps)