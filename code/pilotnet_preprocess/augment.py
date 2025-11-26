# augment.py
import random
import cv2
import numpy as np
from config_data import AUG_BRIGHTNESS, AUG_BLUR, AUG_FLIP

def random_brightness_contrast(img, prob=0.6):
    if not AUG_BRIGHTNESS or random.random() > prob:
        return img
    img = img.astype(np.float32)
    b = random.uniform(-40, 40)
    c = random.uniform(0.8, 1.2)
    img = img * c + b
    return np.clip(img, 0, 255).astype(np.uint8)

def random_blur(img, prob=0.15):
    if not AUG_BLUR or random.random() > prob:
        return img
    k = random.choice([1, 3, 5])
    if k <= 1:
        return img
    return cv2.GaussianBlur(img, (k, k), 0)

def random_flip(img, yaw, prob=0.5):
    if not AUG_FLIP or random.random() > prob:
        return img, yaw
    img = cv2.flip(img, 1)
    yaw = -yaw
    return img, yaw
