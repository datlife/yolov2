"""
DarKNet 19 Architecture
"""
import time
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import GlobalAvgPool2D
from keras.regularizers import l2
from keras.models import Model


def yolo_preprocess_input(x):
    x = x / 255.
    return x


def darknet19(input_size=None, num_classes=1000, include_top=True):
    """
    DarkNet-19 Architecture Definition
    :param input_size:
    :param num_classes:
    :param pretrained_weights:
    :param include_top:
    :return:
    """
    if input_size is None:
        image = Input(shape=(None, None, 3))
    else:
        image = Input(shape=input_size)

    x = conv_block(image, 32, (3, 3))  # << --- Input layer
    x = MaxPool2D(strides=2)(x)

    x = conv_block(x, 64, (3, 3))
    x = MaxPool2D(strides=2)(x)

    x = conv_block(x, 128, (3, 3))
    x = conv_block(x, 64, (1, 1))
    x = conv_block(x, 128, (3, 3))
    x = MaxPool2D(strides=2)(x)

    x = conv_block(x, 256, (3, 3))
    x = conv_block(x, 128, (1, 1))
    x = conv_block(x, 256, (3, 3))
    x = MaxPool2D(strides=2)(x)

    x = conv_block(x, 512, (3, 3))
    x = conv_block(x, 256, (1, 1))
    x = conv_block(x, 512, (3, 3))
    x = conv_block(x, 256, (1, 1))
    x = conv_block(x, 512, (3, 3))
    x = MaxPool2D(strides=2)(x)

    x = conv_block(x, 1024, (3, 3))
    x = conv_block(x, 512, (1, 1))
    x = conv_block(x, 1024, (3, 3))
    x = conv_block(x, 512, (1, 1))
    x = conv_block(x, 1024, (3, 3))    # ---> feature extraction ends here

    if include_top:
        x = Conv2D(num_classes, (1, 1), activation='linear', padding='same')(x)
        x = GlobalAvgPool2D()(x)
        x = Activation(activation='softmax')(x)

    darknet = Model(image, x)

    return darknet


def conv_block(x, filters, kernel_size, name=None):
    """
    Standard YOLOv2 Convolutional Block as suggested in YOLO9000 paper

    Reference: YAD2K github repo
    :param x:
    :param filters:
    :param kernel_size:
    :param kernel_regularizer:
    :return:
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
               use_bias=False, name=name)(x)
    x = BatchNormalization(name=name if name is None else 'batch_norm_%s' % name)(x)
    x = LeakyReLU(alpha=0.1, name=name if name is None else 'leaky_relu_%s' % name)(x)
    return x
