# config.py

import os

# Input resolution cho K210 — QQVGA
WIDTH = 160
HEIGHT = 120
INPUT_SHAPE = (HEIGHT, WIDTH)

# Preprocess settings
CROP_TOP_FRAC = 0.30      # cắt bỏ 30% phía trên ảnh
NORMALIZE_RANGE = "minus1_1"   # ["minus1_1", "0_1"]
TO_GRAYSCALE = True

# Augmentation options
AUG_BRIGHTNESS = True
AUG_BLUR = True
AUG_FLIP = True

# =====================================================
# Dataset paths
# =====================================================
    
# File CSV chứa: file_path,yaw
# CSV_PATH = "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T11-03-25/road_following/data.csv"

# # Thư mục chứa ảnh gốc
# ROOT_DIR = "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T11-03-25/road_following/images"

# # Thư mục lưu ảnh sau khi preprocess (.npy)
# OUT_DIR = "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T11-03-25/dataset_preprocessed"

# thư mục chứa file config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# dataset nằm ở: /home/tv/TiNy_PilotNet/dataset/...
DATASET_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "dataset"))

CSV_PATH = os.path.join(
    DATASET_ROOT,
    "Samples/datasets-T11-03-25/road_following/data.csv"
)

ROOT_DIR = os.path.join(
    DATASET_ROOT,
    "Samples/datasets-T11-03-25/road_following"
)

OUT_DIR = os.path.join(
    DATASET_ROOT,
    "Samples/datasets-T11-03-25/dataset_preprocessed"
)