# scripts/preprocess_folder.py
import os
import sys

# Thêm thư mục cha để import config, preprocess...
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from config import CSV_PATH, ROOT_DIR, OUT_DIR
from preprocess import preprocess_path
from utils import ensure_dir

def main():

    ensure_dir(OUT_DIR)

    # CSV của bạn đã có header: file_path,yaw
    df = pd.read_csv(CSV_PATH)

    for i, row in df.iterrows():
        rel_path = row["file_path"]      # === Quan trọng ===

        img_path = os.path.join(ROOT_DIR, rel_path)

        if not os.path.exists(img_path):
            print("❌ File không tồn tại:", img_path)
            continue

        img = preprocess_path(img_path, augment=False)

        out_path = os.path.join(OUT_DIR, rel_path.replace(".jpg", ".npy"))
        ensure_dir(os.path.dirname(out_path))

        np.save(out_path, img)
        print("Saved:", out_path)

if __name__ == "__main__":
    main()
