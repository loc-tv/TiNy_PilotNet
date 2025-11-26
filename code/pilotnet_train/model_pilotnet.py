# model_pilotnet_tf.py

import tensorflow as tf
from tensorflow import keras  # type: ignore

def PilotNetSmall(input_shape=(120, 160, 1)):
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(24, 5, strides=2, activation='relu')(inputs)
    x = keras.layers.Conv2D(36, 5, strides=2, activation='relu')(x)
    x = keras.layers.Conv2D(48, 5, strides=2, activation='relu')(x)
    x = keras.layers.Conv2D(64, 3, strides=1, activation='relu')(x)
    x = keras.layers.Conv2D(64, 3, strides=1, activation='relu')(x)
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(100, activation='relu')(x)
    x = keras.layers.Dense(50, activation='relu')(x)
    x = keras.layers.Dense(10, activation='relu')(x)
    outputs = keras.layers.Dense(1)(x)

    return keras.Model(inputs, outputs)