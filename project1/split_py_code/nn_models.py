import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import *
import typing

class NN_models():

    def __init__(self, Image_size: typing.Tuple[int], Image_channels=None):
        self.Image_Size = Image_size
        self.Image_Channels = 3 if Image_channels is None else Image_channels
    
    # He's uniform regularizer works better for layers with ReLU activation function
    def He_uniform_4conv_CNN(self) -> keras.models.Sequential:
        assert len(self.Image_Size)==2 and self.Image_Size[0]>0 and self.Image_Size[1]>0, "Image_size must be a tuple of two positive integers!"
        model=Sequential()
        model.add(Conv2D(32,(3,3),activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=42), input_shape=self.Image_Size + (self.Image_Channels, )))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64,(3,3), kernel_initializer=keras.initializers.he_uniform(seed=42), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(128,(3,3), kernel_initializer=keras.initializers.he_uniform(seed=42), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(256,(3,3), kernel_initializer=keras.initializers.he_uniform(seed=42), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512,activation='relu', kernel_regularizer=keras.regularizers.L2(5e-3)))
        model.add(Dense(1,activation='sigmoid'))
        return model
    
    # equivalent to the default layer regularizer at Keras with extra seeding
    def Glorot_uniform_4conv_CNN(self) -> keras.models.Sequential:
        assert len(self.Image_Size)==2 and self.Image_Size[0]>0 and self.Image_Size[1]>0, "Image_size must be a tuple of two positive integers!"
        model=Sequential()
        model.add(Conv2D(32,(3,3),activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=42), input_shape=self.Image_Size + (self.Image_Channels, )))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64,(3,3), kernel_initializer=keras.initializers.glorot_uniform(seed=42), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(128,(3,3), kernel_initializer=keras.initializers.glorot_uniform(seed=42), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(256,(3,3), kernel_initializer=keras.initializers.glorot_uniform(seed=42), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512,activation='relu', kernel_regularizer=keras.regularizers.L2(5e-3)))
        model.add(Dense(1,activation='sigmoid'))
        return model
