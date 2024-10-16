"""
This notebook implements the cross validation code, for a robust evaluation
"""

# %%
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from baseline.train import train_model
from collections import defaultdict 

#from baseline.model import SimpleSegmentationModel
from baseline.linear_model import LinearModel1
from pathlib import Path
from sklearn.metrics import jaccard_score
from baseline.collate import pad_collate
from baseline.dataset import BaselineDataset
from baseline.model import SimpleSegmentationModel
from utils.dummy_model import SimpleSegmentationModelWrapper
#from unet3d.unet3d import UNet



# %% 
"""
Model takes 
X shape (B,T,C,L,H)
Y shape (B, L, H)
"""



def split_dataset(dataset:BaselineDataset, num_folds: int = 5):
    size_fold = len(dataset) // num_folds
    for i in range(num_folds):
        # stop is not reached
        start_val,stop_val = int(i * size_fold), int((i+1) *size_fold)
        indices_val= list(range(start_val, stop_val))
        indices_train = [ind for ind in range(len(dataset)) if ind not in indices_val]
        ds_train = torch.utils.data.Subset(dataset,indices_train)
        ds_val =  torch.utils.data.Subset(dataset,indices_val)
        yield ds_train, ds_val

def train_crossval_loop(
    model_class : torch.nn.modules.module.Module,
    nb_classes: int,
    input_channels: int,
    data_folder: str, 
    max_samples : int = 100,
    num_folds : int = 5,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
):
    """
    Training Loop. No cross validation. Must provide the datasets for training and validation
    Args: 
    - model : should be a PyTorch that accepts (B, T, C, L, H) and returns (B, 20, L, H)
    """
    data_folder=Path(
        data_folder
    )
    assert data_folder.exists()
    dataset = BaselineDataset(data_folder, max_samples=max_samples)
    criterion = nn.CrossEntropyLoss()
    results_folds = defaultdict(list)
    oof_preds = []
    validation_targets = []
    for i, (ds_train, ds_val) in enumerate(split_dataset(dataset,num_folds)):
        model = model_class(input_channels, nb_classes, dim = 3)
        dataloader_train = torch.utils.data.DataLoader(
            ds_train, batch_size=batch_size, collate_fn=pad_collate, shuffle=True
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Move the model to the appropriate device (GPU if available)
        device = torch.device(device)
        model.to(device)

        # Training loop
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0

            for i, (inputs, targets) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
                # Move data to device
                inputs["S2"] = inputs["S2"].to(device)  # Satellite data
                targets = targets.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs["S2"]) 
                # outputs shape B 20 T H W 
                # we want B 20 H W 
                outputs_median_time = torch.median(outputs,2).values
                
                # Loss computation
                loss = criterion(outputs_median_time, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Get the predicted class per pixel (B, H, W)
                preds = torch.argmax(outputs_median_time, dim=1)

                # Move data from GPU/Metal to CPU
                targets = targets.cpu().numpy().flatten()
                preds = preds.cpu().numpy().flatten()

            # Print the loss for this epoch
            epoch_loss = running_loss / len(dataloader_train)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # get validation loss 
        # N, L, H where N is the size of the validation fold 
        model.eval()
        dataloader_val = torch.utils.data.DataLoader(
            ds_val, batch_size=batch_size, collate_fn=pad_collate, shuffle=True
        )
        outputs= []
        # (N , C , L , H)
        print(f"finished training, evaluation over oof")
        targets = []
        for i, (inputs, targets_batch) in tqdm(enumerate(dataloader_val), total=len(dataloader_val)):
            inputs["S2"] = inputs["S2"].to(device)  # Satellite data
            targets_batch= targets_batch.to(device)
            outputs_batch = model(inputs["S2"])
            outputs_batch_median_time  = torch.median(outputs_batch,2).values
            outputs.append(outputs_batch_median_time)
            targets.append(targets_batch)
        outputs_tensor = torch.concat(outputs) # (B, 20, H, W )
        targets = torch.concat(targets)


        # Get the predicted class per pixel (B, H, W)
        preds = torch.argmax(outputs_tensor, dim=1)
        oof_preds.append(preds) # shape (N_fold, H, W), type int 
        validation_targets.append(targets)  # shape (N_fold, H, W), type int  

        # Move data from GPU/Metal to CPU
        targets_flat_npy = targets.cpu().numpy().flatten()
        preds_flat_npy = preds.cpu().numpy().flatten()

        # cv eval loss 
        loss_val = criterion(outputs_tensor, targets)

        # cv eval mean iou 
        mean_iou_val = jaccard_score(preds_flat_npy, targets_flat_npy, average="macro")

        print(f"cv loss {loss_val:.4f} - cv mean_iou {mean_iou_val:.4f}")

        results_folds["fold"].append(i)
        results_folds["loss_val"].append(loss_val)
        results_folds["mean_iou_val"].append(mean_iou_val)
    
    oof_preds_tensor = torch.concat(oof_preds)
    validation_targets_tensor = torch.concat(validation_targets)

    oof_preds_flat_npy = oof_preds_tensor.cpu().numpy().flatten()
    validation_targets_flat_npy = validation_targets_tensor.cpu().numpy().flatten()

    mean_iou_cv = jaccard_score(oof_preds_flat_npy, validation_targets_flat_npy, average="macro")

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
        num_epochs= 1
    )


# %%
