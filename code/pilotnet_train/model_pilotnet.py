# model_pilotnet_tf.py

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def PilotNetSmall(input_shape=(120, 160, 1)):
    inputs = Input(shape=input_shape)

    x = Conv2D(24, 5, strides=2, activation='relu')(inputs)
    x = Conv2D(36, 5, strides=2, activation='relu')(x)
    x = Conv2D(48, 5, strides=2, activation='relu')(x)
    x = Conv2D(64, 3, strides=1, activation='relu')(x)
    x = Conv2D(64, 3, strides=1, activation='relu')(x)

    x = Flatten()(x)

    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    outputs = Dense(1)(x)

    return Model(inputs, outputs)
