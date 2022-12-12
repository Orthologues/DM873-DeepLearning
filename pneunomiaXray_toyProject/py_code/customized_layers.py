"""
Written by Jiawei Zhao on 12th of December, 2022 to implemented customized Keras layers 
"""

import tensorflow as tf
from keras.initializers import HeUniform, GlorotUniform
import keras.backend as K 
from typing import *


"""
This class is a very rudimentary customized layer that is equivalent to "keras.layers.Conv2D"
"""
class MyConv2D(tf.keras.layers.Layer):
    """
    >>> MyConv2D(filters=32).compute_output_shape(input_shape=(32, 224, 224, 3))
    (222, 222, 32)
    """

    def __init__(self, filters: int, kernel_initializer:str = "he_uniform", kernel_size: Tuple[int] = (3, 3), activation:str='relu', **kwargs):
        # assertions for input params
        assert isinstance(filters, int) and filters > 0
        assert isinstance(kernel_initializer, str)
        assert kernel_initializer in {"he_uniform", "glorot_uniform"}
        assert len(kernel_size)==2 and kernel_size[0]>1 and kernel_size[1]>1
        assert isinstance(activation, str) and activation in {"relu", "sigmoid"}
        # name initialization
        super(MyConv2D, self).__init__(name="MyConv2D")
        # class attribute assignment
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        # call the constructor of the parent class of MyConv (tf.keras.layers.Layer)
        super(MyConv2D, self).__init__(**kwargs)
    
    """
    Overrides .get_config() method to avoid an error that would halt training after the first epoch
    """
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer
        })
        return config

    def build(self, input_shape):
        # assertions for input params
        assert len(input_shape) == 4
        # update the kernels of the 2D convolutional layer
        w_init = HeUniform(seed=42) if self.kernel_initializer=="he_uniform" else GlorotUniform(seed=42)
        self.w = tf.Variable(name='weight',
                             initial_value=w_init(shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters)),
                             trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name='bias',
                             initial_value=b_init(shape=(input_shape[1] - self.kernel_size[0] + 1, input_shape[2] - self.kernel_size[1] + 1) + (self.filters, )),
                             trainable=True)

    def call(self, inputs):
        output = K.conv2d(inputs, self.w, data_format="channels_last")
        return tf.nn.relu(output + self.b) if self.activation=="relu" else tf.nn.sigmoid(output + self.b)

    def compute_output_shape(self, input_shape):
        return (input_shape[1] - self.kernel_size[0] + 1, input_shape[2] - self.kernel_size[1] + 1) + (self.filters, )
        

"""
This class is a very rudimentary customized layer that is equivalent to "keras.layers.MaxPooling2D"
"""
class MyMaxPool2D(tf.keras.layers.Layer):
    """
    >>> tuple(MyMaxPool2D().compute_output_shape(input_shape=(32, 222, 222, 64)))
    (32, 111.0, 111.0, 64)
    """

    def __init__(self, pool_size: Tuple[int]=(2, 2), strides: Tuple[int]=(2, 2), **kwargs):
        # assertions for input params
        assert len(pool_size)==2 and pool_size[0]>1 and pool_size[1]>1
        assert len(strides)==2 and strides[0]>1 and strides[1]>1
        # name initialization
        super(MyMaxPool2D, self).__init__(name="MyMaxPool2D")
        self.pool_size = pool_size
        self.strides = strides
        super(MyMaxPool2D, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
        })
        return config
    
    def call(self, inputs):
        return K.pool2d(inputs, pool_size=tuple(self.pool_size), strides=tuple(self.strides))

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        if K.image_data_format() == 'channels_first':
            shape[2] /= 2
            shape[3] /= 2
        else:
            shape[1] /= 2
            shape[2] /= 2
        return tuple(shape)


"""
This class is a very rudimentary customized layer that is equivalent to "keras.layers.Dense"
"""
class MyDense(tf.keras.layers.Layer):
    """
    >>> tuple(MyDense(units=1, activation="sigmoid").compute_output_shape(input_shape=(784, 128)))
    (784, 1)
    """

    def __init__(self, kernel_regularizer: str="l2", units:int=1, activation:str='relu', **kwargs):
        # assertions for input params
        assert isinstance(units, int) and units > 0
        assert isinstance(kernel_regularizer, str)
        assert kernel_regularizer in {"l1", "l2", "l1l2"}
        assert isinstance(activation, str) and activation in {"relu", "sigmoid"}
        # name initialization
        super(MyDense, self).__init__(name="MyDense") 
        # class attribute assignment
        self.units = units
        self.kernel_regularizer = kernel_regularizer
        self.activation = activation
        super(MyDense, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'kernel_regularizer': self.kernel_regularizer,
            'activation': self.activation
        })
        return config

    def build(self, input_shape):
        # assertions for input params
        assert len(input_shape) > 1
        # class attribute assignment 
        input_shape = tuple(input_shape)
        self.w = self.add_weight(name='weight',  
                        shape=(input_shape[-1], self.units),
                        initializer='uniform',
                        regularizer=self.kernel_regularizer,
                        trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name='bias',
                             initial_value=b_init(shape=(self.units, )),
                             trainable=True)
    
    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b) if self.activation=="relu" else tf.nn.sigmoid(tf.matmul(inputs, self.w) + self.b)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
