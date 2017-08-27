import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import LeakyReLU
from keras.layers.merge import concatenate
from keras.models import Model

from models.zoo.mobilenet import _depthwise_conv_block, _conv_block


def MobileYOLOv2(img_size=(608, 608, 3), num_classes=90, num_anchors=5, alpha=1.0, depth_multiplier=1):
    """
    Original YOLOv2 Architecture
    :param img_size:
    :param num_classes:
    :param num_anchors:
    :param alpha:
    :param depth_multiplier:
    :return:
    """

    img_input = Input(shape=img_size)

    # Feature extractor Layers (DarkNet-19)
    x = _conv_block(img_input, 32, alpha, strides=(2, 2))

    x = _depthwise_conv_block(x, 64,  alpha, depth_multiplier, block_id=1)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=2, strides=(2, 2))
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=4, strides=(2, 2))
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=6, strides=(2, 2))
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    i = x  # fine-grained layer
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=12, strides=(2, 2))
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    # Object Detector layers
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=14)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=15)
    x = reroute(i, x, stride=2)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=16)

    # Output layers
    x = Conv2D(filters=num_anchors * (5 + num_classes), kernel_size=(1, 1), activation='linear')(x)

    model = Model(inputs=img_input, outputs=x)
    return model


def conv_block(x, filters, kernel_size, kernel_regularizer=None):
    """
    YOLOv2 Convolutional Block as suggested in YOLO9000 paper

    Reference: YAD2K repo
    :param x:
    :param filters:
    :param kernel_size:
    :param kernel_regularizer:
    :return:
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, kernel_regularizer=kernel_regularizer, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def reroute(x1, x2, stride=2):
    x = _conv_block(x1, 32, 1.0, kernel=(1, 1), name='conv2')
    x = Lambda(lambda x: tf.space_to_depth(x, block_size=stride),
               lambda shape: [shape[0], shape[1] / stride, shape[2] / stride, stride * stride * shape[-1]] if shape[
                   1] else
               [shape[0], None, None, stride * stride * shape[-1]],
               name='space_to_depth_x2')(x)
    x = concatenate([x, x2])
    return x

if __name__ == "__main__":
    yolomobile = MobileYOLOv2()
    yolomobile.summary()
