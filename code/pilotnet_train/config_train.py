# config_train_tf.py

# Danh sách dataset giữ nguyên format của bạn
DATASETS = [
    (
        "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T10-30-25/road_following/data.csv",
        "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T10-30-25/dataset_preprocessed"
    ),
    (
        "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T11-03-25/road_following/data.csv",
        "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T11-03-25/dataset_preprocessed"
    ),
    (
        "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T11-03to06-25/road_following/data.csv",
        "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T11-03to06-25/dataset_preprocessed"
    ),
    (
        "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T11-07-25/road_following/data.csv",
        "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T11-07-25/dataset_preprocessed"
    ),
    (
        "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T11-09-25/road_following/data.csv",
        "/home/tv/TiNy_PilotNet/dataset/Samples/datasets-T11-09-25/dataset_preprocessed"
    ),
]

# Training params
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

MODEL_OUT = "/home/tv/TiNy_PilotNet/model/pilotnet_small_keras.h5"
