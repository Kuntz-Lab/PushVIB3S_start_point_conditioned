import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from bps import get_random_basis, encode_pcd_with_bps, normalize

from utils import *

class PushDataset(Dataset):
    def __init__(self, data_dir="processed_data/", bps_dir="bps/", use_directional_bps=True, deterministic_farthest_point_sampling=False):
        self.data_dir = data_dir
        self.file_list = []
        self.max_dist = 0.0
        self.use_directional_bps = use_directional_bps

        self.num_points_per_pc = 512  # Number of points to sample per point cloud
        self.deterministic_farthest_point_sampling = deterministic_farthest_point_sampling
        
        print(f"Loading data from {data_dir}")
        print(f"Using directional BPS: {use_directional_bps}")

        # Try to load existing basis, if not create and save a new one
        basis_path = os.path.join(bps_dir, 'bps_basis_r1.0_n128.npy')

        if os.path.exists(basis_path):
            # Kept on CPU: __getitem__ runs in DataLoader worker processes, and producing
            # CUDA tensors there requires CUDA IPC to hand them back to the main process,
            # which is fragile (see "Producer process has been terminated..." warnings).
            self.my_basis = torch.tensor(np.load(basis_path, allow_pickle=True), dtype=torch.float, device='cpu')
            print("Loaded existing BPS basis")
        else:
            raise FileNotFoundError(f"BPS basis file not found at {basis_path}. Please create the basis file before proceeding. You can use bps.py")
        
        # Get list of all pickle files in directory
        print(f"Getting list of files in {data_dir}")
        self.file_list = find_pickle_files(data_dir)
                
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        device = torch.device('cpu')
        # Load pickle file
        with open(self.file_list[idx], 'rb') as f:
            data = pickle.load(f)

        item = {}
            
        start_pc = data['start_pointcloud']
        goal_pc = data['goal_pointcloud']

        if 'start_position' in data and 'displacement' in data:
            # grab the ground truth result and pass it along in the dataset for use in evaluation and visualization
            start_point = data['start_position']
            displacement = data['displacement']

            if start_point.shape[0] == 3:
                start_point = np.expand_dims(start_point, axis=0)  # Add batch dimension
            if displacement.shape[0] == 3:
                displacement = np.expand_dims(displacement, axis=0)  # Add batch dimension
            
            # ensure the type of start_point and displacement is float32 for consistency
            start_point = start_point.astype(np.float32)
            displacement = displacement.astype(np.float32)

            item["start_point"] = start_point
            item["displacement"] = displacement

        if start_pc.shape[0] > self.num_points_per_pc:
            start_pc = farthest_point_sampling(start_pc, self.num_points_per_pc, use_internally_random_seed=(not self.deterministic_farthest_point_sampling))
        if goal_pc.shape[0] > self.num_points_per_pc:
            goal_pc = farthest_point_sampling(goal_pc, self.num_points_per_pc, use_internally_random_seed=(not self.deterministic_farthest_point_sampling))

            # # Visualization for debugging
            # visualize_prediction(start_pc, goal_pc, start_point, displacement)

        # normalize point cloud to fit within the BPS sphere
        start_pc_normalized = np.expand_dims(np.copy(start_pc), axis=0)  # Add batch dimension
        goal_pc_normalized = np.expand_dims(np.copy(goal_pc), axis=0)  # Add batch dimension
        start_pc_normalized, x_mean, x_max = normalize(start_pc_normalized, return_scalers=True)
        start_pc_normalized = start_pc_normalized.squeeze(0)  # Remove batch dimension

        # Goal pc already has batch dimension
        goal_pc_normalized, _, _ = normalize(goal_pc_normalized, known_scalers=True, x_mean=x_mean, x_max=x_max, return_scalers=True)
        goal_pc_normalized = goal_pc_normalized.squeeze(0)  # Remove batch dimension

        # Create tensors for start and goal point clouds
        start_pc_normalized = start_pc_normalized.to(device) if isinstance(start_pc_normalized, torch.Tensor) else torch.FloatTensor(start_pc_normalized).to(device)
        goal_pc_normalized = goal_pc_normalized.to(device) if isinstance(goal_pc_normalized, torch.Tensor) else torch.FloatTensor(goal_pc_normalized).to(device)

        # Normalize start point
        start_point_normalized = ((item["start_point"] - x_mean) / x_max).astype(np.float32)
        displacement_normalized = (item["displacement"] / x_max).astype(np.float32)
        
        # Use the class attribute for directional BPS
        if self.use_directional_bps:
            # Encode with directions
            start_bps_dist, start_bps_dir = encode_pcd_with_bps(start_pc_normalized, self.my_basis, return_directions=True)
            goal_bps_dist, goal_bps_dir = encode_pcd_with_bps(goal_pc_normalized, self.my_basis, return_directions=True)
            
            # Reshape direction vectors correctly
            # Direction vectors are [batch, n_points, 3], we need to reshape to [batch, n_points*3]
            start_bps_dir_flat = start_bps_dir.reshape(start_bps_dir.shape[0], -1)
            goal_bps_dir_flat = goal_bps_dir.reshape(goal_bps_dir.shape[0], -1)
            
            # Concatenate distances and directions along feature dimension
            start_bps = torch.cat([start_bps_dist, start_bps_dir_flat], dim=1)
            goal_bps = torch.cat([goal_bps_dist, goal_bps_dir_flat], dim=1)
        else:
            # Original distance-only encoding
            start_bps = encode_pcd_with_bps(start_pc_normalized, self.my_basis)
            goal_bps = encode_pcd_with_bps(goal_pc_normalized, self.my_basis)
            
        
        item["start_point_normalized"] = start_point_normalized
        item["displacement_normalized"] = displacement_normalized
        item["start_bps"] = start_bps
        item["goal_bps"] = goal_bps
        item["start_pc"] = start_pc_normalized
        item["goal_pc"] = goal_pc_normalized
        item["start_pc_unnormalized"] = start_pc  # Keep the original unnormalized point clouds
        item["goal_pc_unnormalized"] = goal_pc
        item["x_mean"] = x_mean # These are for use in denormalization
        item["x_max"] = x_max
        item["file_name"] = self.file_list[idx]  # Store the file name for reference
            
        return item

