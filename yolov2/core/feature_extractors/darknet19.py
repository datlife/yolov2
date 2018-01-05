"""
DarKNet 19 Architecture
"""
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import GlobalAvgPool2D
from keras.regularizers import l2
from ..custom_layers import Preprocessor


def yolov2_preprocess_func(inputs):
    inputs = inputs / 255.
    return inputs


def darknet19(inputs, num_classes=1000, include_top=False):
    """
    DarkNet-19 Architecture Definition

    Args:
      inputs:
      num_classes:
      include_top:

    Returns:
      x: model definition
      fine_grained_layers - a list of fine_grained layers (for detection)
    """
    pass_through_layers = []
    with tf.name_scope('DarkNet19'):
        inputs = Preprocessor(yolov2_preprocess_func, name='preprocessor')(inputs)
        x = conv_block(inputs, 32, (3, 3), name="Conv_1")
        x = MaxPool2D(strides=2)(x)

        x = conv_block(x, 64, (3, 3), name="Conv_2")
        x = MaxPool2D(strides=2)(x)

        x = conv_block(x, 128, (3, 3), name="Conv_3")
        x = conv_block(x, 64, (1, 1), name="Conv_4")
        x = conv_block(x, 128, (3, 3), name="Conv_5")
        x = MaxPool2D(strides=2)(x)

        x = conv_block(x, 256, (3, 3), name="Conv_6")
        x = conv_block(x, 128, (1, 1), name="Conv_7")
        x = conv_block(x, 256, (3, 3), name="Conv_8")
        x = MaxPool2D(strides=2)(x)

        x = conv_block(x, 512, (3, 3), name="Conv_9")
        x = conv_block(x, 256, (1, 1), name="Conv_10")
        x = conv_block(x, 512, (3, 3), name="Conv_11")
        x = conv_block(x, 256, (1, 1), name="Conv_12")
        x = conv_block(x, 512, (3, 3), name="Conv_13")
        with tf.name_scope('PassThroughLayers'):
            pass_through_layers.append(x)
        x = MaxPool2D(strides=2)(x)

        x = conv_block(x, 1024, (3, 3), name="Conv_14")
        x = conv_block(x, 512, (1, 1),  name="Conv_15")
        x = conv_block(x, 1024, (3, 3), name="Conv_16")
        x = conv_block(x, 512, (1, 1),  name="Conv_17")
        x = conv_block(x, 1024, (3, 3), name="Conv_18")    # ---> feature extraction ends here

        if include_top:
            x = Conv2D(num_classes, (1, 1), activation='linear', padding='same')(x)
            x = GlobalAvgPool2D()(x)
            x = Activation(activation='softmax')(x)
            x = Model(inputs, x)

    return x, pass_through_layers


def conv_block(x, filters, kernel_size, name=None):
    """
        Standard YOLOv2 Convolutional Block as suggested in YOLO9000 paper
    """
    scope = None
    if name:
        name_block, idx = name.split('_')
        scope = "".join([name_block, "Block_", idx])

    with tf.name_scope(scope, 'ConvBlock'):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   padding='same',
                   use_bias=False,
                   name=name)(x)
        x = BatchNormalization(name=name if name is None else '%s_BatchNorm' % name)(x)
        x = LeakyReLU(alpha=0.1, name=name if name is None else '%s_LeakyReLU' % name)(x)
        return x

