"""
Implementation of YOLOv2 in Keras
"""
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Lambda
from keras.layers import MaxPool2D
from keras.layers.merge import concatenate


def YOLOv2(img_size=(608, 608, 3), num_classes=90, num_anchors=5, kernel_regularizer=None):
    """
    Original YOLOv2 Architecture
    :param img_size:
    :param num_classes:
    :param num_anchors:
    :param kernel_regularizer:
    :return:
    """

    img_input = Input(shape=img_size)

    # Feature extractor Layers (DarkNet-19)
    x = conv_block(img_input, 32, (3, 3), kernel_regularizer=kernel_regularizer)
    x = MaxPool2D()(x)
    x = conv_block(x, 64, (3, 3), kernel_regularizer=kernel_regularizer)
    x = MaxPool2D()(x)

    x = conv_block(x, 128, (3, 3), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 64,  (1, 1), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 128, (3, 3), kernel_regularizer=kernel_regularizer)
    x = MaxPool2D()(x)

    x = conv_block(x, 256, (3, 3), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 128, (1, 1), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 256, (3, 3), kernel_regularizer=kernel_regularizer)
    x = MaxPool2D()(x)

    x = conv_block(x, 512, (3, 3), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 256, (1, 1), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 512, (3, 3), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 256, (1, 1), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 512, (3, 3), kernel_regularizer=kernel_regularizer)
    i = x                           # fine-grained layer
    x = MaxPool2D()(x)

    x = conv_block(x, 1024, (3, 3), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 512,  (1, 1), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 1024, (3, 3), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 512,  (1, 1), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 1024, (3, 3), kernel_regularizer=kernel_regularizer)

    # Object Detector layers
    x = conv_block(x, 1024, (3, 3), kernel_regularizer=kernel_regularizer)
    x = conv_block(x, 1024, (3, 3), kernel_regularizer=kernel_regularizer)
    x = reroute(i, x, stride=2)
    x = conv_block(x, 1024, (3, 3), kernel_regularizer=kernel_regularizer)

    # Output layers
    x = Conv2D(filters=num_anchors*(5 + num_classes), kernel_size=(1, 1), name='yolov2',
               activation='linear', kernel_regularizer=kernel_regularizer)(x)

    model = Model(inputs=img_input, outputs=x)
    return model


def conv_block(x, filters, kernel_size, kernel_regularizer=None):
    """
    Standard YOLOv2 Convolutional Block as suggested in YOLO9000 paper

    Reference: YAD2K github repo
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
    x = conv_block(x1, 64, (1, 1))
    x = Lambda(lambda x: tf.space_to_depth(x, block_size=stride),
               lambda shape: [shape[0], shape[1] / stride, shape[2] / stride, stride*stride*shape[-1]] if shape[1] else
                             [shape[0], None, None,  stride*stride*shape[-1]],
               name='space_to_depth_x2')(x)
    x = concatenate([x, x2])
    return x


# TEST CASES:
if __name__ == "__main__":

    # Test 1 - COCO Dataset
    yolov2 = YOLOv2(img_size=(608, 608, 3), num_classes=90, num_anchors=5)
    yolov2.summary()

    # Test 2 - VOC Dataset
    yolov2 = YOLOv2(img_size=(416, 416, 3), num_classes=20, num_anchors=5)
    yolov2.summary()

    # Test 3 - New Data set, variable image sizes
    yolov2 = YOLOv2(img_size=(None, None, 3), num_classes=15, num_anchors=5)
    yolov2.summary()

