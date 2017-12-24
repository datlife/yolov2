"""
DarKNet 19 Architecture
"""
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


def conv_block(x, filters, kernel_size, name=None):
    """
        Standard YOLOv2 Convolutional Block as suggested in YOLO9000 paper

    :param x:
    :param filters:
    :param kernel_size:
    :param name:
    :return:
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
               use_bias=False, name=name)(x)
    x = BatchNormalization(name=name if name is None else 'batch_norm_%s' % name)(x)
    x = LeakyReLU(alpha=0.1, name=name if name is None else 'leaky_relu_%s' % name)(x)
    return x


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
    fine_grained_layers = []

    x = conv_block(inputs, 32, (3, 3))  # << --- Input layer
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
    fine_grained_layers.append(x)
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
        x = Model(inputs, x)

    return x, fine_grained_layers




