# cross_val_xgboost.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss, jaccard_score
import xgboost as xgb
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# Import the split_dataset function from cross_val.py
from baseline.cross_val import split_dataset

# Assuming these modules are available based on your provided code
from baseline.collate import pad_collate
from baseline.dataset import BaselineDataset

def prepare_data_subset(dataset_subset, device='cpu'):
    """
    Prepares a subset of the dataset for XGBoost by computing the median across the time dimension
    and reshaping the data into (num_samples, num_features).

    Args:
        dataset_subset (torch.utils.data.Subset): Subset of the BaselineDataset.
        device (str): Device to load tensors ('cpu' or 'cuda').

    Returns:
        features (np.ndarray): Feature matrix for XGBoost.
        labels (np.ndarray): Corresponding labels.
    """
    dataloader = DataLoader(dataset_subset, batch_size=4, shuffle=False, collate_fn=pad_collate)
    features = []
    labels = []

    for inputs, targets in tqdm(dataloader, desc="Preparing data for subset"):
        # inputs["S2"] shape: (B, T, C, L, H)
        # Compute median across time (dim=1)
        s2_median = torch.median(inputs["S2"], dim=1).values  # shape: (B, C, L, H)

        # Reshape to (B, C, L*H)
        B, C, L, H = s2_median.shape
        s2_median = s2_median.view(B, C, L * H)  # shape: (B, C, L*H)

        # Transpose to (B, L*H, C) to have each pixel as a sample
        s2_median = s2_median.permute(0, 2, 1)  # shape: (B, L*H, C)

        # Convert to NumPy and reshape
        features_batch = s2_median.cpu().numpy().reshape(-1, C)  # shape: (B * L * H, C)
        features.append(features_batch)

        # Flatten targets to (B * L * H,)
        targets_batch = targets.view(targets.size(0), -1).cpu().numpy().reshape(-1)
        labels.append(targets_batch)

    # Concatenate all batches
    features = np.concatenate(features, axis=0)  # shape: (total_pixels, C)
    labels = np.concatenate(labels, axis=0)      # shape: (total_pixels,)

    return features, labels

def train_xgboost_crossval(
    data_folder: str,
    max_samples: int = 100,
    num_folds: int = 5,
    learning_rate: float = 1e-3,
    verbose: bool = False,
):
    """
    Trains an XGBoost classifier using cross-validation.

    Args:
        data_folder (str): Path to the data folder.
        max_samples (int): Maximum number of samples to use.
        num_folds (int): Number of cross-validation folds.
        learning_rate (float): Learning rate for XGBoost.
        verbose (bool): Whether to print detailed logs.

    Returns:
        final_clf (xgb.XGBClassifier): Trained XGBoost model on the entire dataset.
        results_folds (defaultdict): Dictionary containing validation loss and IoU for each fold.
        mean_iou_cv (float): Overall mean IoU across all folds.
    """
    data_folder = Path(data_folder)
    assert data_folder.exists(), f"Data folder {data_folder} does not exist."

    # Initialize the dataset
    dataset = BaselineDataset(data_folder, max_samples=max_samples)

    # Determine the number of classes
    # Assuming that the dataset can provide the number of classes
    # Alternatively, set nb_classes manually if known
    nb_classes = 20  # Replace with the actual number of classes if different

    # Initialize results storage
    results_folds = defaultdict(list)
    oof_preds = []
    validation_targets = []

    # Generate train and validation splits using split_dataset
    for fold, (ds_train, ds_val) in enumerate(split_dataset(dataset, num_folds=num_folds)):
        print(f"\n--- Fold {fold + 1}/{num_folds} ---")

        # Prepare training data
        X_train, y_train = prepare_data_subset(ds_train)
        print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")

        # Prepare validation data
        X_val, y_val = prepare_data_subset(ds_val)
        print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")

        # Initialize XGBoost classifier
        clf = xgb.XGBClassifier(
            objective='multi:softprob',  # For multi-class classification with probabilities
            num_class=nb_classes,
            eval_metric='mlogloss',
            use_label_encoder=False,
            learning_rate=learning_rate,
            n_estimators=100,  # You can adjust the number of trees
            max_depth=6,        # You can adjust the tree depth
            verbosity=0,
            n_jobs=-1
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

    # Optionally, train a final model on the entire dataset
    print("\nTraining final model on the entire dataset...")
    X_full, y_full = prepare_data_subset(dataset)
    final_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=nb_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        learning_rate=learning_rate,
        n_estimators=100,
        max_depth=6,
        verbosity=0,
        n_jobs=-1
    )
    final_clf.fit(X_full, y_full)
    print("Final model training complete.")

    return final_clf, results_folds, mean_iou_cv

def main():
    # Configuration
    data_folder = "path_to_your_data_folder"  # Replace with your actual data path
    max_samples = 100  # Adjust based on your dataset size and memory constraints
    num_folds = 5
    learning_rate = 1e-3  # Adjust as needed

    # Train XGBoost with cross-validation
    final_clf, results_folds, mean_iou_cv = train_xgboost_crossval(
        data_folder=data_folder,
        max_samples=max_samples,
        num_folds=num_folds,
        learning_rate=learning_rate,
        verbose=True
    )

    # Optionally, save the final trained model
    final_clf.save_model("xgboost_final_model.json")
    print("\nFinal model saved as 'xgboost_final_model.json'.")

    # Display cross-validation results
    print("\nCross-Validation Results:")
    for fold in results_folds["fold"]:
        loss = results_folds["loss_val"][fold - 1]
        iou = results_folds["mean_iou_val"][fold - 1]
        print(f"Fold {fold}: Loss = {loss:.4f}, Mean IoU = {iou:.4f}")

    print(f"\nOverall Cross-Validation Mean IoU: {mean_iou_cv:.4f}")

if __name__ == "__main__":
    main()
