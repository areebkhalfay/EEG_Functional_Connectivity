"""
Handling dataset preparation for PyTorch Training.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class EEGSceneDataset(Dataset):
    def __init__(self, X, y):
        """
        X: (N, window_size, C) -> converted to (N, C, T) for PyTorch
        y: (N,)
        """
        self.X = torch.from_numpy(X).float().permute(0, 2, 1) 
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_loaders_for_loso(per_subject_data, test_subj, batch_size=64):
    train_X_list, train_y_list = [], []
    brainRegions = None

    for subj_id, data in per_subject_data.items():
        if subj_id == test_subj:
            brainRegions = data["Brain Regions"]
            continue
        train_X_list.append(data["X"])
        train_y_list.append(data["y"])

    X_train = np.concatenate(train_X_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)

    X_test = per_subject_data[test_subj]["X"]
    y_test = per_subject_data[test_subj]["y"]

    # Handle class imbalance
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    
    pos_count = max(pos_count, 1)
    neg_count = max(neg_count, 1)
    
    weight_for_pos = 1.0 / pos_count
    weight_for_neg = 1.0 / neg_count
    
    sample_weights = np.where(y_train == 1, weight_for_pos, weight_for_neg).astype(np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = EEGSceneDataset(X_train, y_train)
    test_dataset  = EEGSceneDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, brainRegions