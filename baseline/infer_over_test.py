# %%
from pathlib import Path
import torch 
import torch.nn as nn 
import pandas as pd 
from tqdm import tqdm 
from baseline.submission_tools import masks_to_str
from baseline.dataset import OutputDataset
from baseline.collate import pad_collate 
from utils.model_io import load_full_model



def model_inference(
    model : nn.Module, 
    data_folder: Path,
    batch_size: int = 1,
    device: str = "cpu",
    dataset_class = OutputDataset,
    max_samples: int |None = None

) -> None:
    """
    Training pipeline.
    Args : 
    - model: the already-trained model. 
    """
    # Create data loader
    dataset = dataset_class(data_folder, max_samples = max_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=pad_collate, shuffle=False
    )

    # Load the saved model
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
            preds = torch.argmax(outputs, dim=1).cpu()
            preds_str = masks_to_str(preds)
            res.append([patch_id.item(),preds_str[0]])
    return pd.DataFrame(res,columns=['ID','MASKS'])

if __name__ == "__main__":
    FILE_PATH = Path("outputs/SimpleSegmentationModelWrapper_10-18_22-33")
    assert FILE_PATH.exists()
    model = load_full_model(file_path=FILE_PATH)
    data_folder = Path("DATA-mini")
    result = model_inference(model,data_folder)
    

# %%

# %%
