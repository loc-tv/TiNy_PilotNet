# dataset_pilotnet.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf


class PilotNetNPYDatasetTF:
    def __init__(self, csv_path, root_npy):
        self.df = pd.read_csv(csv_path)
        self.root = root_npy

    def __len__(self):
        return len(self.df)

    def generator(self):
        for i in range(len(self.df)):
            row = self.df.iloc[i]

            npy_file = row["file_path"].replace(".jpg", ".npy")
            full_path = os.path.join(self.root, npy_file)

            img = np.load(full_path).astype(np.float32)

            # Numpy HWC cho Keras
            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            yaw = np.float32(row["yaw"])
            yield img, yaw


class PilotNetConcatDatasetTF:
    @staticmethod
    def load_from_config(dataset_list):
        datasets = []
        for csv_path, root in dataset_list:
            ds = PilotNetNPYDatasetTF(csv_path, root)
            datasets.append(ds)
            print(f"[INFO] Loaded dataset: {csv_path}  ({len(ds)} samples)")
        return datasets
