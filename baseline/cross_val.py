"""
This notebook implements the cross validation code, for a robust evaluation
"""

# %%
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from baseline.train import train_model
from collections import defaultdict 
from baseline.collate import pad_collate
from baseline.model import SimpleSegmentationModel
from datetime import datetime
from pathlib import Path
from sklearn.metrics import jaccard_score
from baseline.collate import pad_collate
from baseline.dataset import BaselineDataset
from baseline.model import SimpleSegmentationModel
from utils.dummy_model import SimpleSegmentationModelWrapper
from unet3d.unet3d import UNet
from utils.model_io import save_full_model
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


def eval_loop(
        model:nn.Module,
        dataloader_val:  torch.utils.data.DataLoader,
        device : str = "cuda",
        debug : bool = False):
    model.eval()
    outputs= []
    targets = []
    # (N , C , L , H)
    with torch.no_grad():
        for i, (inputs, targets_batch) in tqdm(enumerate(dataloader_val), total=len(dataloader_val)):
            inputs["S2"] = inputs["S2"].to(device)  # Satellite data
            targets_batch= targets_batch.to(device)
            outputs_batch = model(inputs["S2"], debug = debug)
            # outputs should be of shape B 20 H W 
            outputs.append(outputs_batch.cpu())
            targets.append(targets_batch.cpu())
            if debug: 
                print(f"{inputs['S2'].shape=}")
                print(f"outputs batch shape {outputs_batch.shape}")
        if debug : 
            print(f"len outputs {outputs.__len__()}", f"{outputs[0].shape=}")
        outputs_tensor = torch.concat(outputs).cpu() # (B, 20, H, W )
        targets = torch.concat(targets).cpu()
    return outputs_tensor, targets 

def train_crossval_loop(
    model_class : torch.nn.modules.module.Module,
    data_folder: str,
    num_folds : int = 5,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    get_validation_loss_during_training : bool = True,
    dataset_class = BaselineDataset,
    max_samples : int = None,
    debug : bool = False,
    weights_criterion : torch.Tensor|None= None,
    **model_kwargs  
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
    dataset = dataset_class(data_folder, max_samples=max_samples)
    if weights_criterion is not None:
        criterion = nn.CrossEntropyLoss(weight=weights_criterion)
    else:
        criterion = nn.CrossEntropyLoss()
    oof_preds = []
    validation_targets = []
    results_training = {
        "oof_preds" : oof_preds,
        "validation_targets" : validation_targets,
        "training_metrics" : {}
    }

    for fold_nbr, (ds_train, ds_val) in enumerate(split_dataset(dataset,num_folds)):
        model = model_class(**model_kwargs)
        dataloader_train = torch.utils.data.DataLoader(
            ds_train, batch_size=batch_size, collate_fn=pad_collate, shuffle=True
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Move the model to the appropriate device (GPU if available)
        device = torch.device(device)
        model.to(device)
        results_per_epoch = defaultdict(list)
        fold_preds = None # predictions, the ones for which the val iou was smallest 
        # Training loop
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            print(f"fold {fold_nbr} - epoch {epoch} - training batch [1/{len(dataloader_train)}]...")
            for batch_nbr, (inputs, targets) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):

                # Move data to device
                inputs_batch = inputs["S2"].to(device)  # Satellite data
                targets = targets.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # outputs should be of shape (B, 20, H ,W)
                outputs = model(inputs_batch, debug = debug) 
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Print the loss for this epoch
            epoch_loss = running_loss / len(dataloader_train)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            results_per_epoch["loss"].append(epoch_loss)
            results_per_epoch["epoch"].append(epoch)

            # get the score for this epoch
            if get_validation_loss_during_training:
                if epoch == num_epochs -1 or  (fold_nbr == 0 and epoch % 2 == 0): 
                    print(f"...running inference for validation loss and iou...")
                    dataloader_val = torch.utils.data.DataLoader(
                        ds_val, batch_size=batch_size, collate_fn=pad_collate, shuffle=False
                    )
                    outputs_tensor, targets = eval_loop(model, dataloader_val,device, debug =debug )
                    preds = torch.argmax(outputs_tensor, dim=1)
                    targets_flat_npy = targets.numpy().flatten()
                    preds_flat_npy = preds.numpy().flatten()
                    mean_iou_val_epoch= jaccard_score(preds_flat_npy, targets_flat_npy, average="macro")
                    print(f"Fold {fold_nbr}, Epoch {epoch} : Val IOU {mean_iou_val_epoch:.3f}, Val Loss {epoch_loss:.3f} ")
                    results_per_epoch["iou"].append(mean_iou_val_epoch)
                    if mean_iou_val_epoch  ==  max(results_per_epoch["iou"]):
                        fold_preds = preds
                    """
                    If the score for this epoch is better than the last, then keep the preds for this epoch
                    """
        # get validation loss 
        # N, H, W where N is the size of the validation fold 
        results_training["oof_preds"].append(fold_preds) # shape (N_fold, H, W), type int 
        results_training["validation_targets"].append(targets)  # shape (N_fold, H, W), type int  
        results_training["training_metrics"][f"fold_{fold_nbr}"] = results_per_epoch
        file_path_results_training = f"/kaggle/working/training_metrics_{model_class.__name__}_{datetime.now().strftime('%m-%d_%H-%M')}_fold{fold_nbr}.pkl"
        with open(file_path_results_training, 'wb') as f:
            pickle.dump(results_training, f)
    oof_preds_tensor = torch.concat(oof_preds)
    validation_targets_tensor = torch.concat(validation_targets)

    oof_preds_flat_npy = oof_preds_tensor.cpu().numpy().flatten()
    validation_targets_flat_npy = validation_targets_tensor.cpu().numpy().flatten()

    mean_iou_cv = jaccard_score(oof_preds_flat_npy, validation_targets_flat_npy, average="macro")

    print(f"IOU full CV : {mean_iou_cv}")

    return model, results_training, mean_iou_cv
# %%
if __name__ == "__main__" : 
    # # how to split accurately
    # # grouped split 
    # folds = 5 
    # batch_size = 10 
    # input_channels = 10 
    # nb_classes = 20
    # model_class = SimpleSegmentationModelWrapper

    # model, results_training, mean_iou_cv = train_crossval_loop(
    #     model_class =model_class,
    #     data_folder="DATA-mini",
    #     batch_size=1,
    #     num_epochs= 3,
    #     # model kwargs 
    #     in_channels = 10,
    #     out_channels = 20,
    # )

    # save_full_model(model, f"outputs/{model_class.__name__}_{datetime.now().strftime(f'%m-%d_%H-%M')}")
    folds = 5 
    batch_size = 10 
    input_channels = 10 
    nb_classes = 20
    model_class = SimpleSegmentationModelWrapper
    data_folder = Path("DATA-mini")
    assert data_folder.exists()
    model, results_training, mean_iou_cv = train_crossval_loop(
        model_class = model_class,
        data_folder=data_folder,
        batch_size=4,
        num_epochs= 2,
        num_folds = 2,
        device= "cpu", 
        max_samples = 40,
        # model kwargs below: 
        in_channels = 10,
        out_channels = 20,
        # dim = 3
    )

# %%
