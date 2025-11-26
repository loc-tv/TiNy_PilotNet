# preprocess.py
import numpy as np
from config_data import CROP_TOP_FRAC, TO_GRAYSCALE
from image_ops import load_rgb, crop_top, resize_qqvga, to_gray, normalize
from augment import random_brightness_contrast, random_blur, random_flip

def preprocess_path(path, augment=True):
    img = load_rgb(path)
    return preprocess_image(img, augment)

def preprocess_image(img, augment=True, yaw=None):
    img = crop_top(img, CROP_TOP_FRAC)
    img = resize_qqvga(img)

    if augment:
        img = random_brightness_contrast(img)
        img = random_blur(img)
        if yaw is not None:
            img, yaw = random_flip(img, yaw)

    if TO_GRAYSCALE:
        img = to_gray(img)

    img = normalize(img)

    # # convert (H,W,C) → (C,H,W)
    # img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)


    if yaw is None:
        return img.astype(np.float32)
    return img.astype(np.float32), np.float32(yaw)
