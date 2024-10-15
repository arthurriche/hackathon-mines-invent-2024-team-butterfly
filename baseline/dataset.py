"""
Baseline Pytorch Dataset
"""

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import pandas as pd


class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, folder: Path):
        super(BaselineDataset, self).__init__()
        self.folder = folder

        # Get metadata
        print("Reading patch metadata ...")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        # Process 'dates-S2' column to convert date strings to datetime objects
        print("Processing 'dates-S2' ...")
        # Adjust the lambda function based on the actual structure of 'dates-S2'
        self.meta_patch['dates-S2'] = self.meta_patch['dates-S2'].apply(
            lambda x: pd.to_datetime(list(x.values()), format='%Y%m%d') if isinstance(x, dict) else pd.to_datetime(x, format='%Y%m%d')
        )
        print("Date processing complete.")

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

        # If you have other modalities, add them as fields of the `data` dict ...
        # data["radar"] = ...
        dates = self.meta_patch['dates-S2'].iloc[item]
        if isinstance(dates, pd.DatetimeIndex):
            # If dates is a DatetimeIndex, convert to list of strings
            dates_list = dates.strftime('%Y-%m-%d').tolist()
        elif isinstance(dates, pd.Series):
            # If dates is a Series, convert each entry to string
            dates_list = dates.dt.strftime('%Y-%m-%d').tolist()
        elif isinstance(dates, pd.Timestamp):
            # If dates is a single Timestamp, wrap in a list
            dates_list = [dates.strftime('%Y-%m-%d')]
        elif isinstance(dates, list):
            # If dates is a list, convert each date to string
            dates_list = [date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date) for date in dates]
        else:
            # Handle unexpected types
            print(f"Warning: Unexpected type for dates: {type(dates)}. Attempting to convert to string.")
            dates_list = [str(dates)]

        # Assign dates to data dictionary
        data['dates'] = dates_list  # List of date strings


        # Open and prepare targets
        target = np.load(
            os.path.join(self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch))
        )
        target = torch.from_numpy(target[0].astype(int))

        return data, target
