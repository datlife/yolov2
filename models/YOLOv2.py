"""
YOLOv2 Meta-Architecture

        images --> feature extractor --> feature map --> detector --> output feature map

In this file, we define three different detector:
   * Original YOLOv2 Detector
   * MobileNet-type detector
   * DenseNet-type detector

-----------------------
Example usage: This code will define pretrained YOLOv2 on COCO Dataset (80 classes)

from models.YOLOv2 import YOLOv2
from models.loss import custom_loss
from models.FeatureExtractor import FeatureExtractor

# Define YOLOv2 with Darknet-19 as feature extractor and use 'yolov2' as detector
darknet = FeatureExtractor(img_size=(608, 608, 3), model='yolov2')

yolo    = YOLOv2(num_classes       = 80,
                 anchors           = np.array(ANCHORS),
                 feature_extractor = darknet,
                 detector          = 'yolov2')

# Compile model
model = yolo.model.compile('adam',loss=custom_loss)

"""
from zoo.mobilenet import _depthwise_conv_block, relu6
from zoo.darknet19 import conv_block
from keras.layers import Lambda, Conv2D, BatchNormalization, Activation
from keras.layers.merge import concatenate
import tensorflow as tf
from keras.models import Model
from keras.regularizers import l2
FINE_GRAINED_LAYERS = {'yolov2': 'leaky_re_lu_13',
                       'mobilenet': 'conv_pw_11_relu',
                       'densenet': 'leaky_re_lu_136'}


def yolov2_detector(feature_extractor, num_classes, num_anchors, fine_grained_layers):
    """
    Original YOLOv2 Implementation
    :param feature_extractor:
    :param num_classes:
    :param num_anchors:

    :return:
    """

    inputs = feature_extractor.model.output
    i = feature_extractor.model.get_layer(fine_grained_layers).output

    x = conv_block(inputs, 1024, (3, 3))
    x = conv_block(x, 1024, (3, 3))
    x2 = x

    # Reroute
    x = conv_block(i, 64, (1, 1))
    x = Lambda(lambda x: tf.space_to_depth(x, block_size=2),
               lambda shape: [shape[0], shape[1] / 2, shape[2] / 2, 2 * 2 * shape[-1]] if shape[1] else
               [shape[0], None, None, 2 * 2 * shape[-1]],
               name='space_to_depth_x2')(x)

    x = concatenate([x, x2])
    x = conv_block(x, 1024, (3, 3))
    x = Conv2D(num_anchors * (num_classes + 5), (1, 1), name='yolov2', kernel_regularizer=l2(5e-4))(x)

    return x


def mobile_detector(feature_extractor, num_classes, num_anchors, fine_grained_layers):
    """
    Mobile Detector Implementation
    :param feature_extractor:
    :param num_classes:
    :param num_anchors:
    :param fine_grained_layers

    :return:
    """
    inputs = feature_extractor.model.output
    i = feature_extractor.model.get_layer(fine_grained_layers).output

    x = _depthwise_conv_block(inputs, 1024, 1.0, block_id=14)
    x = _depthwise_conv_block(x, 1024, 1.0, block_id=15)
    x2 = x

    # Reroute
    x = Conv2D(64, (1, 1), padding='same', use_bias=False, strides=(1, 1))(i)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    x = Lambda(lambda x: tf.space_to_depth(x, block_size=2),
               lambda shape: [shape[0], shape[1] / 2, shape[2] / 2, 2 * 2 * shape[-1]] if shape[1] else
               [shape[0], None, None, 2 * 2 * shape[-1]],
               name='space_to_depth_x2')(x)
    x = concatenate([x, x2])

    x = _depthwise_conv_block(x, 1024, 1.0, block_id=16)
    x = Conv2D(num_anchors * (num_classes + 5), (1, 1))(x)

    return x


def densenet_detector(feature_extractor, num_classes, num_anchors, fine_grained_layers):
    """
    DenseNet Implementation - For Now I keeped this architecture the same as YOLov2 due to a few incompatibilities
    :param feature_extractor:
    :param num_classes:
    :param num_anchors:
    :param fine_grained_layers

    :return:
    """
    inputs = feature_extractor.model.output
    i = feature_extractor.model.get_layer(fine_grained_layers).output

    x = conv_block(inputs, 1024, (3, 3), name='detector_1')
    x = conv_block(x, 1024, (3, 3), name='detector_2')
    x2 = x

    # Reroute
    x = conv_block(i, 64, (1, 1), name='detector_3')
    x = Lambda(lambda x: tf.space_to_depth(x, block_size=2),
               lambda shape: [shape[0], shape[1] / 2, shape[2] / 2, 2 * 2 * shape[-1]] if shape[1] else
               [shape[0], None, None, 2 * 2 * shape[-1]],
               name='space_to_depth_x2')(x)

    x = concatenate([x, x2], name='concat_2')
    x = conv_block(x, 1024, (3, 3), name='detector_4')
    x = Conv2D(num_anchors * (num_classes + 5), (1, 1), name='yolov2-densenet')(x)

    return x


DETECTOR = {'yolov2': yolov2_detector,
            'mobilenet': mobile_detector,
            'densenet': densenet_detector}


class YOLOv2(object):
    def __init__(self,
                 is_training,
                 num_classes,
                 anchors,
                 feature_extractor,
                 detector
                 ):
        """
        YOLOv2 Meta-Architecture

        images --> feature extractor --> feature map --> detector --> output feature map

        :param num_classes:
        :param anchors:
        :param is_training:
        :param feature_extractor:
        :param detector:
        """

        self.num_classes       = num_classes
        self.anchors           = anchors
        self._is_training      = is_training
        self.feature_extractor = feature_extractor

        self.fine_grained_layers = FINE_GRAINED_LAYERS[feature_extractor.name]
        self.detector = DETECTOR[detector](feature_extractor, num_classes, len(anchors), self.fine_grained_layers)

        # YOLOv2 Model
        self.model = Model(inputs=feature_extractor.model.input,
                           outputs=self.detector)

    def post_process(self):
        pass

    def loss(self):
        pass


def reroute(x1, x2, stride=2):
    x = conv_block(x1, 64, (1, 1))
    x = Lambda(lambda x: tf.space_to_depth(x, block_size=stride),
               lambda shape: [shape[0], shape[1] / stride, shape[2] / stride, stride * stride * shape[-1]] if shape[
                   1] else
               [shape[0], None, None, stride * stride * shape[-1]],
               name='space_to_depth_x2')(x)
    x = concatenate([x, x2])

    return x
