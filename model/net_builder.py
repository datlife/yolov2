"""
Util to build a convolutional block easier
"""
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU


def conv_block(x, filters=32, kernel_size=(3, 3), padding='same'):
    """
    Yolov2 Convolutional Block [Conv --> Batch Norm --> LeakyReLU]
    :param x: 
    :param filters: 
    :param kernel_size: 
    :param padding: 
    :return: 
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, use_bias=False, padding=padding)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

