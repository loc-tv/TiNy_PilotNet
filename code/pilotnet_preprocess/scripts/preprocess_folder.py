import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from config_data import DATASETS
from preprocess import preprocess_path
from utils import ensure_dir


def preprocess_one(csv_path, root_dir, out_dir):
    ensure_dir(out_dir)

    df = pd.read_csv(csv_path)

    for i, row in df.iterrows():
        rel = row["file_path"]
        img_path = os.path.join(root_dir, rel)

        if not os.path.exists(img_path):
            print("❌ Missing:", img_path)
            continue

        img = preprocess_path(img_path, augment=False)

        out_path = os.path.join(out_dir, rel.replace(".jpg", ".npy"))
        ensure_dir(os.path.dirname(out_path))

        np.save(out_path, img)
        print("Saved:", out_path)


def main():
    for d in DATASETS:
        print("\n======================================")
        print("📁 PROCESSING DATASET:", d["name"])
        print("======================================")

        preprocess_one(
            csv_path=d["csv"],
            root_dir=d["root"],
            out_dir=d["out"]
        )


if __name__ == "__main__":
    main()
