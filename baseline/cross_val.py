"""
This notebook implements the cross validation code, for a robust evaluation
"""

# %%
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict 

from baseline.linear_model import LinearModel1
from pathlib import Path
from sklearn.metrics import jaccard_score
from baseline.collate import pad_collate
from baseline.dataset import BaselineDataset
from utils.dummy_model import SimpleSegmentationModelWrapper
#from unet3d.unet3d import UNet



# %% 
"""
Model takes 
X shape (B,T,C,L,H)
Y shape (B, L, H)
"""



def split_dataset(dataset: BaselineDataset, num_folds: int = 5):
    size_fold = len(dataset) // num_folds
    for i in range(num_folds):
        start_val, stop_val = int(i * size_fold), int((i + 1) * size_fold)
        indices_val = list(range(start_val, stop_val))
        indices_train = [ind for ind in range(len(dataset)) if ind not in indices_val]
        ds_train = torch.utils.data.Subset(dataset, indices_train)
        ds_val = torch.utils.data.Subset(dataset, indices_val)
        yield ds_train, ds_val


def train_crossval_loop(
    model_class: torch.nn.Module,
    nb_classes: int,
    input_channels: int,
    data_folder: str, 
    max_samples: int = 10,
    num_folds: int = 5,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
):
    """
    Cross-validation loop for training.
    Args: 
    - model_class: The model class to instantiate.
    - nb_classes: Number of segmentation classes.
    - input_channels: Number of input channels.
    - data_folder: Path to the dataset.
    """
    data_folder = Path(data_folder)
    assert data_folder.exists()
    
    dataset = BaselineDataset(data_folder, max_samples=max_samples)
    criterion = nn.CrossEntropyLoss()
    results_folds = defaultdict(list)
    oof_preds = []
    validation_targets = []

    for i, (ds_train, ds_val) in enumerate(split_dataset(dataset, num_folds)):
        # Instantiate the model with the correct parameters
        model = model_class(
            C=input_channels,
            W=128,
            H=128,
            L1=25,  # Adjust L1 as needed
            nb_classes=nb_classes
        )
        model.to(device)

        dataloader_train = torch.utils.data.DataLoader(
            ds_train, batch_size=batch_size, collate_fn=pad_collate, shuffle=True
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0

            for i, (inputs, targets) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
                # Move data to device
                inputs["S2"] = inputs["S2"].to(device)  # Satellite data
                targets = targets.to(device).float()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs["S2"]) 
                outputs_median_time = torch.median(outputs, dim=1).values  # Median over time dimension
                targets = targets.float().mean(dim=1).long()  # Ensure correct target shape

                loss = criterion(outputs_median_time, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Get the predicted class per pixel
                preds = torch.argmax(outputs_median_time, dim=1)

                # Move data from GPU/Metal to CPU for metrics
                targets = targets.cpu().numpy().flatten()
                preds = preds.cpu().numpy().flatten()

            epoch_loss = running_loss / len(dataloader_train)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        dataloader_val = torch.utils.data.DataLoader(
            ds_val, batch_size=batch_size, collate_fn=pad_collate, shuffle=False
        )
        outputs = []
        targets = []

        for i, (inputs, targets_batch) in tqdm(enumerate(dataloader_val), total=len(dataloader_val)):
            inputs["S2"] = inputs["S2"].to(device)  # Satellite data
            targets_batch = targets_batch.to(device)
            outputs_batch = model(inputs["S2"])
            outputs_batch_median_time = torch.median(outputs_batch, dim=1).values  # Median over time dimension
            outputs.append(outputs_batch_median_time)
            targets.append(targets_batch)

        outputs_tensor = torch.cat(outputs)  # (B, nb_classes, H, W)
        targets = torch.cat(targets)

        # Get the predicted class per pixel
        preds = torch.argmax(outputs_tensor, dim=1)
        oof_preds.append(preds)
        validation_targets.append(targets)

        # Compute validation metrics
        targets_flat = targets.cpu().numpy().flatten()
        preds_flat = preds.cpu().numpy().flatten()

        # Ensure both arrays have the same length
        if len(targets_flat) == len(preds_flat):
            mean_iou_val = jaccard_score(preds_flat, targets_flat, average="macro")
        else:
            raise ValueError(f"Inconsistent number of samples: {len(targets_flat)} and {len(preds_flat)}")
        
        loss_val = criterion(outputs_tensor, targets.float())
        mean_iou_val = jaccard_score(preds_flat, targets_flat, average="macro")

        print(f"Fold {i}, Validation Loss: {loss_val:.4f}, Mean IOU: {mean_iou_val:.4f}")

        results_folds["fold"].append(i)
        results_folds["loss_val"].append(loss_val.item())
        results_folds["mean_iou_val"].append(mean_iou_val)
    
    oof_preds_tensor = torch.cat(oof_preds)
    validation_targets_tensor = torch.cat(validation_targets)

    mean_iou_cv = jaccard_score(oof_preds_tensor.cpu().numpy().flatten(), 
                                validation_targets_tensor.cpu().numpy().flatten(), 
                                average="macro")

    print("Training complete.")
    return model, results_folds, mean_iou_cv
# %%
if __name__ == "__main__" : 
    # how to split accurately
    # grouped split 
    folds = 5 
    batch_size = 10 

    # lets try the cross val function above

    input_channels = 10 
    nb_classes = 20

    model, results_folds, mean_iou_cv = train_crossval_loop(
        model_class = LinearModel1,
        nb_classes=20,
        input_channels= 10,
        batch_size=1,
        num_epochs= 5,
        data_folder="/Users/ludoviclepic/Desktop/Capgemini/hackathon-mines-invent-2024-team-butterfly/TRAIN"
    )


# %%
