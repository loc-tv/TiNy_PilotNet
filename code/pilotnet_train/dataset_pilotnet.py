import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class PilotNetNPYDataset(Dataset):
    def __init__(self, csv_path, root_npy):
        self.df = pd.read_csv(csv_path)
        self.root = root_npy

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        npy_file = row["file_path"].replace(".jpg", ".npy")
        full_path = os.path.join(self.root, npy_file)

        img = np.load(full_path).astype(np.float32)

# đảm bảo đúng định dạng NCHW
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.transpose(img, (2, 0, 1))

        yaw = np.float32(row["yaw"])
        return torch.tensor(img), torch.tensor(yaw)
    

