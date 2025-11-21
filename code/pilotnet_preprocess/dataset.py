# dataset.py
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from preprocess import preprocess_image
from image_ops import load_rgb

class PilotNetDataset(Dataset):
    def __init__(self, csv_file, root_dir="", augment=False):
        self.df = pd.read_csv(csv_file, header=None, names=["file","yaw"])
        self.root = root_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.root, row["file"])
        yaw = float(row["yaw"])

        img = load_rgb(path)
        img, yaw = preprocess_image(img, augment=self.augment, yaw=yaw)

        return torch.tensor(img), torch.tensor([yaw])
