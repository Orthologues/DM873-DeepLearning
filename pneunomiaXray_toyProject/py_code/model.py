"""
Written by Jiawei Zhao on 12th of December, 2022 to implemented customized Keras layers 
"""

from keras.models import Sequential
from customized_layers import *
from tensorflow import keras
from keras.layers import BatchNormalization, Flatten, Dropout

def create_self_CNN_Model(image_size: tuple, image_channels=3) -> tf.keras.Sequential:
    # assertions for input params
    assert len(image_size)==2 and image_size[0]>0 and image_size[1]>0, "Image_size must be a tuple of two positive integers!"
    assert image_channels in {1, 3, 4}
    # construct the sequential model
    model=Sequential()
    model.add(MyConv2D(filters=32, kernel_initializer=keras.initializers.he_uniform(seed=42), activation='relu'))
    model.add(BatchNormalization())
    model.add(MyMaxPool2D())
    model.add(MyConv2D(filters=64, kernel_initializer=keras.initializers.he_uniform(seed=42), activation='relu'))
    model.add(BatchNormalization())
    model.add(MyMaxPool2D())
    model.add(MyConv2D(filters=128, kernel_initializer=keras.initializers.he_uniform(seed=42), activation='relu'))
    model.add(BatchNormalization())
    model.add(MyMaxPool2D())
    model.add(MyConv2D(filters=256, kernel_initializer=keras.initializers.he_uniform(seed=42), activation='relu'))
    model.add(BatchNormalization())
    model.add(MyMaxPool2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(MyDense(units=512, kernel_regularizer=keras.regularizers.L2(5e-3)))
    model.add(MyDense(units=1, activation='sigmoid'))
    return model