# config.py

import os

# =====================================================
# Global preprocess settings
# =====================================================

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
    
# =====================================================
# Dataset paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# thư mục dataset gốc: /home/tv/TiNy_PilotNet/dataset
DATASET_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "dataset")
)

# --- Khai báo nhiều dataset ở đây ---
DATASETS = [
    {
        "name": "datasets-T10-30-25",
        "csv": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T10-30-25/road_following/data.csv"),
        "root": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T10-30-25/road_following"),
        "out": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T10-30-25/dataset_preprocessed")
    },

    {
        "name": "datasets-T11-03-25",
        "csv": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T11-03-25/road_following/data.csv"),
        "root": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T11-03-25/road_following"),
        "out": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T11-03-25/dataset_preprocessed")
    },

    {
        "name": "datasets-T11-03to06-25",
        "csv": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T11-03to06-25/road_following/data.csv"),
        "root": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T11-03to06-25/road_following"),
        "out": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T11-03to06-25/dataset_preprocessed")
    },
    
    {
        "name": "datasets-T11-07-25",
        "csv": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T11-07-25/road_following/data.csv"),
        "root": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T11-07-25/road_following"),
        "out": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T11-07-25/dataset_preprocessed")
    },
    
    {
        "name": "datasets-T11-09-25",
        "csv": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T11-09-25/road_following/data.csv"),
        "root": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T11-09-25/road_following"),
        "out": os.path.join(DATASET_ROOT, 
                "Samples/datasets-T11-09-25/dataset_preprocessed")
    },
]