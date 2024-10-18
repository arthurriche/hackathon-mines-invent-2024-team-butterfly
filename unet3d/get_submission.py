import torch
from torch.utils.data import DataLoader
from torch import nn
import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import torch
from unet3d.unet3d import UNet
from unet3d.collate import pad_collate
import tqdm

#Create an a dataset to get both the patches and the index for the test data.

class OutputDataset(torch.utils.data.Dataset):
    def __init__(self, folder: Path):
        super(OutputDataset, self).__init__()
        self.folder = folder

        # Get metadata
        print("Reading patch metadata ...")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        print("Done.")

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index
        print("Dataset ready.")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        id_patch = self.id_patches[item]

        # Open and prepare satellite data into T x C x H x W arrays
        path_patch = os.path.join(self.folder, "DATA_S2", "S2_{}.npy".format(id_patch))
        data = np.load(path_patch).astype(np.float32)
        data = {"S2": torch.from_numpy(data)}
        #Get the id for output submission
        data['ID'] = id_patch

        return data

class OutputNoCloudDataset(torch.utils.data.Dataset):
    def __init__(self, folder: Path):
        super(OutputNoCloudDataset, self).__init__()
        self.folder = folder

        # Get metadata
        print("Reading patch metadata ...")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        print("Done.")

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index
        print("Dataset ready.")

    def __len__(self) -> int:
        return self.len

    @staticmethod
    def remove_clouds(sample,n_channels=10,q_diff=.8,cloud_perc=.5):

        median = torch.median(sample,axis=0).values

        diff = torch.sqrt((sample-median.view(1,sample.size(1),sample.size(2),sample.size(3))) ** 2)

        quantile = torch.quantile(diff
                                .transpose(1,0)
                                .reshape(n_channels,-1),q_diff,dim=-1)

        mask = diff>quantile.view(1,n_channels,1,1)
        is_not_cloudy = ~(torch.max(torch.sum(torch.sum(mask,dim=-1),dim=-1) / (sample.size(-1) ** 2) > cloud_perc,dim=-1).values)
        
        return sample[is_not_cloudy,...]
    
    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        id_patch = self.id_patches[item]

        # Open and prepare satellite data into T x C x H x W arrays
        path_patch = os.path.join(self.folder, "DATA_S2", "S2_{}.npy".format(id_patch))
        data = np.load(path_patch).astype(np.float32)
        data = {"S2": self.remove_clouds(torch.from_numpy(data))}
        #Get the id for output submission
        data['ID'] = id_patch

        return data

def masks_to_str(predictions: np.ndarray) -> list[str]:
    """
    Convert the

    Args:
        predictions (np.ndarray): predictions as a 3D batch (B, H, W)

    Returns:
        list[str]: a list of B strings, each string is a flattened stringified prediction mask
    """
    return [" ".join(f"{x}" for x in np.ravel(x)) for x in predictions]

def eval_model(
    data_folder: Path,
    model_file: str,
    batch_size: int = 1,
    device: str = "cpu",

) -> None:
    """
    Training pipeline.
    """
    # Create data loader
    dataset = OutputNoCloudDataset(data_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=pad_collate, shuffle=False
    )

    # Load the saved model
    in_channels = 10
    out_channels = 20
    dim = 3
    model = UNet(in_channels=in_channels,out_channels=out_channels,dim=dim)
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.to(device)

    # Set the model in evaluation mode
    model.eval()

    # 3. Evaluate the Model on Test Samples
    # Disable gradient computation for evaluation
    res = []
    with torch.no_grad():
        for i, (inputs) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Move data to device
            inputs["S2"] = inputs["S2"].to(device)  # Satellite data
            patch_id = inputs['ID']
            # Forward pass through the model
            outputs = model(inputs['S2'])
            outputs_median_time = torch.median(outputs,2).values
            preds = torch.argmax(outputs_median_time, dim=1).cpu()
            preds_str = masks_to_str(preds)
            res.append([patch_id.item(),preds_str[0]])
    return pd.DataFrame(res,columns=['ID','MASKS'])

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

preds = eval_model(data_folder=Path('/kaggle/input/data-challenge-invent-mines-2024/DATA/DATA/TEST'),
                   model_file='/kaggle/input/unet3d/pytorch/default/1/unet3d.pt',
                   device=DEVICE)
output_path = 'submissions_1_unet3d.csv'
preds.to_csv(output_path,index=False)