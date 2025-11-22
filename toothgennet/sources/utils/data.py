import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle

cate_to_cate_idx = {
    'U1': 0, 'U2': 1, 'U3': 2, 'U4': 3, 'U5': 4, 'U6': 5, 'U7': 6, 'U8': 7,
    'L1': 8, 'L2': 9, 'L3': 10, 'L4': 11, 'L5': 12, 'L6': 13, 'L7': 14, 'L8': 15,
}

class ToothNet15KPC_SingleCategory(Dataset):
    def __init__(self, pcd_pkl_dict, split='train', tooth_category='U1', n_sample_points=2048, use_random_sampling=True):
        self.pcd_pkl_dict = pcd_pkl_dict
        self.split = split
        self.tooth_category = tooth_category
        self.n_sample_points = n_sample_points
        self.use_random_sampling = use_random_sampling
        
        # Check if the split and category exist in the dictionary
        if self.split not in self.pcd_pkl_dict:
            raise ValueError(f"Split '{self.split}' not found in dataset dictionary.")
        if self.tooth_category not in self.pcd_pkl_dict[self.split]:
            raise ValueError(f"Category '{self.tooth_category}' not found in split '{self.split}'.")
            
        self.all_stems = self.pcd_pkl_dict[self.split][self.tooth_category]['stems']
        self.all_points = self.pcd_pkl_dict[self.split][self.tooth_category]['points']
    
    def __len__(self):
        return len(self.all_stems)

    def __getitem__(self, index):
        if self.use_random_sampling:
            sample_point_indexes = np.random.choice(self.all_points[index].shape[0], self.n_sample_points)
        else:
            # Ensure we don't go out of bounds if n_sample_points > available points
            # For PointFlow, usually points are pre-sampled or sufficient, but good to be safe or just take first N
            # The original code used np.arange(self.n_sample_points), assuming sufficient points.
            # We'll stick to original logic but might need safety check if points < n_sample_points
            if self.all_points[index].shape[0] < self.n_sample_points:
                 # Fallback to random choice with replacement if not enough points
                 sample_point_indexes = np.random.choice(self.all_points[index].shape[0], self.n_sample_points, replace=True)
            else:
                sample_point_indexes = np.arange(self.n_sample_points)
                
        return {
            'index': index,
            'points': self.all_points[index],
            'sample_points': self.all_points[index][sample_point_indexes],
            'cate_name': self.tooth_category,
            'cate_index': cate_to_cate_idx.get(self.tooth_category, -1),
            'stem': self.all_stems[index],
        }

def get_data_loaders(dataset_path, tooth_category='U1', n_sample_points=2048, train_batch_size=32, test_batch_size=32, num_workers=0):
    """
    Creates data loaders for training, validation, and testing.
    
    Args:
        dataset_path (str or Path): Path to the pickle file containing the dataset.
        tooth_category (str): Tooth category to load (e.g., 'U1').
        n_sample_points (int): Number of points to sample per point cloud.
        train_batch_size (int): Batch size for training.
        test_batch_size (int): Batch size for testing/validation.
        num_workers (int): Number of worker processes for data loading.
        
    Returns:
        dict: A dictionary containing 'train_loader', 'train_unshuffle_loader', 'val_loader', 'test_loader'.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        
    with open(dataset_path, 'rb') as f:
        pcd_pkl_dict = pickle.load(f)

    tr_dataset = ToothNet15KPC_SingleCategory(pcd_pkl_dict, split='train', tooth_category=tooth_category, n_sample_points=n_sample_points, use_random_sampling=True)
    te_dataset = ToothNet15KPC_SingleCategory(pcd_pkl_dict, split='test', tooth_category=tooth_category, n_sample_points=n_sample_points, use_random_sampling=False)
    val_dataset = ToothNet15KPC_SingleCategory(pcd_pkl_dict, split='val', tooth_category=tooth_category, n_sample_points=n_sample_points, use_random_sampling=False)
    
    train_loader = DataLoader(tr_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    train_unshuffle_loader = DataLoader(tr_dataset, batch_size=train_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(te_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    data_loaders = {
        'test_loader': test_loader,
        'val_loader': val_loader,
        'train_loader': train_loader,
        'train_unshuffle_loader': train_unshuffle_loader,
    }
    return data_loaders
