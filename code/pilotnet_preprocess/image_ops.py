# image_ops.py
import cv2
import numpy as np
from config_data import WIDTH, HEIGHT, NORMALIZE_RANGE

def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def crop_top(img, frac):
    h = img.shape[0]
    y0 = int(h * frac)
    return img[y0:, :, :]

def resize_qqvga(img):
    return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

def to_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return g[..., None]  # shape (H, W, 1)

def normalize(img):
    if NORMALIZE_RANGE == "0_1":
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32) / 127.5 - 1.0
