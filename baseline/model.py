import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, jaccard_score
import xgboost as xgb
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# Assuming these modules are available based on your provided code
from baseline.collate import pad_collate
from baseline.dataset import BaselineDataset

def prepare_data(dataset, device='cpu'):
    """
    Prepares the dataset for XGBoost by computing the median across the time dimension
    and reshaping the data into (num_samples, num_features).

    Args:
        dataset (BaselineDataset): Your custom dataset.
        device (str): Device to load tensors ('cpu' or 'cuda').

    Returns:
        features (np.ndarray): Feature matrix for XGBoost.
        labels (np.ndarray): Corresponding labels.
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)
    features = []
    labels = []
    
    for inputs, targets in tqdm(dataloader, desc="Preparing data"):
        # inputs["S2"] shape: (1, T, C, L, H)
        # Compute median across time
        s2_median = torch.median(inputs["S2"], dim=1).values  # shape: (1, C, L, H)
        
        # Reshape to (C, L*H)
        C, L, H = s2_median.shape[1:]
        s2_median = s2_median.view(1, C, L * H).squeeze(0)  # shape: (C, L*H)
        
        # Transpose to (L*H, C) to have each pixel as a sample
        s2_median = s2_median.permute(1, 0)  # shape: (L*H, C)
        
        # Convert to NumPy
        features.append(s2_median.cpu().numpy())  # List of arrays
        
        # Flatten targets to (L*H,)
        targets = targets.view(-1).cpu().numpy()
        labels.append(targets)
    
    # Concatenate all batches
    features = np.concatenate(features, axis=0)  # shape: (total_pixels, C)
    labels = np.concatenate(labels, axis=0)      # shape: (total_pixels,)
    
    return features, labels

def train_xgboost_crossval(features, labels, num_folds=5, num_classes=None):
    """
    Trains an XGBoost classifier using cross-validation.

    Args:
        features (np.ndarray): Feature matrix.
        labels (np.ndarray): Target labels.
        num_folds (int): Number of cross-validation folds.
        num_classes (int, optional): Number of unique classes. If None, it will be inferred.

    Returns:
        clf (xgb.XGBClassifier): Trained XGBoost model on the entire dataset.
        results_folds (defaultdict): Dictionary containing validation loss and IoU for each fold.
        mean_iou_cv (float): Overall mean IoU across all folds.
    """
    if num_classes is None:
        num_classes = len(np.unique(labels))
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    results_folds = defaultdict(list)
    oof_preds = []
    validation_targets = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        print(f"\n--- Fold {fold + 1}/{num_folds} ---")
        
        # Initialize XGBoost classifier
        clf = xgb.XGBClassifier(
            objective='multi:softprob',  # For cross-entropy loss
            num_class=num_classes,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=-1,
            verbosity=0
        )
        
        # Fit the model
        clf.fit(X_train, y_train)
        
        # Predict probabilities and classes on validation set
        y_pred_proba = clf.predict_proba(X_val)
        y_pred = clf.predict(X_val)
        
        # Calculate cross-entropy loss
        loss_val = log_loss(y_val, y_pred_proba)
        
        # Calculate mean IoU
        mean_iou_val = jaccard_score(y_val, y_pred, average='macro')
        
        print(f"Validation Log Loss: {loss_val:.4f}")
        print(f"Validation Mean IoU: {mean_iou_val:.4f}")
        
        # Store results
        results_folds["fold"].append(fold + 1)
        results_folds["loss_val"].append(loss_val)
        results_folds["mean_iou_val"].append(mean_iou_val)
        
        oof_preds.append(y_pred)
        validation_targets.append(y_val)
    
    # Concatenate out-of-fold predictions and targets
    oof_preds = np.concatenate(oof_preds)
    validation_targets = np.concatenate(validation_targets)
    
    # Overall mean IoU
    mean_iou_cv = jaccard_score(validation_targets, oof_preds, average='macro')
    print(f"\nCross-Validation Mean IoU: {mean_iou_cv:.4f}")
    
    # Train final model on entire dataset
    final_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0
    )
    final_clf.fit(features, labels)
    
    return final_clf, results_folds, mean_iou_cv

def main():
    # Configuration
    data_folder = "path_to_your_data_folder"  # Replace with your actual data path
    max_samples = 100  # Adjust based on your dataset size and memory constraints
    num_folds = 5
    batch_size = 1  # Must be 1 for proper reshaping in prepare_data
    device = "cpu"  # Change to "cuda" if GPU is available and desired
    
    # Initialize dataset
    dataset = BaselineDataset(Path(data_folder), max_samples=max_samples)
    
    # Prepare data
    features, labels = prepare_data(dataset, device=device)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Train with cross-validation
    clf, results_folds, mean_iou_cv = train_xgboost_crossval(
        features,
        labels,
        num_folds=num_folds
    )
    
    # Optionally, save the trained model
    clf.save_model("xgboost_final_model.json")
    
    # Display cross-validation results
    print("\nCross-Validation Results:")
    for fold in results_folds["fold"]:
        loss = results_folds["loss_val"][fold - 1]
        iou = results_folds["mean_iou_val"][fold - 1]
        print(f"Fold {fold}: Loss = {loss:.4f}, Mean IoU = {iou:.4f}")
    
    print(f"\nOverall Cross-Validation Mean IoU: {mean_iou_cv:.4f}")

if __name__ == "__main__":
    main()


import torch.nn as nn


class SimpleSegmentationModel(nn.Module):
    def __init__(self, input_channels: int, nb_classes: int):
        super(SimpleSegmentationModel, self).__init__()

        # A very basic architecture: Encoder + Decoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, nb_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Input x shape: (B, Channels, H, W)
        x = self.encoder(x)
        x = self.decoder(x)
        # Output x shape: (B, Classes, H, W)
        return x
