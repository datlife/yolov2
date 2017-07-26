"""
Implementation of YOLOv2 Architecture on Keras

Few notes:
   - Its feature extractor is still DarkNet-19 (a VGG-ish CNN)
   - Its object detector structure is still the same. Instead of using regular CNN, however, I used MobileNet-like CNN
     in oder to reduce the number of parameters (from 57M --> 26.5M parameters )
"""
from keras.layers.merge import concatenate
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.regularizers import l2
from keras.models import Model
from model.darknet19 import darknet19
from model.net_builder import _depthwise_conv_block, conv_block


class MobileYolo(object):
    """
    Yolov2 Meta-Architecture
    """
    def __init__(self, feature_extractor=None, num_anchors=9, num_classes=31, fine_grain_layer=43):
        """
        :param feature_extractor: A high-level CNN Classifier. One can plug and update an new feature extractor
                    e.g. :  Darknet19 (YOLOv2), MobileNet, ResNet-50
                    **NOTE** Update the SHRINK_FACTOR accordingly (number of max-pooling layer)s
        :param num_anchors: int  
                    - number of anchors   
        :param num_classes: int 
                   - number of classes in training data
        """
        self.n_anchors = num_anchors
        self.n_classes = num_classes
        self.model = self._construct_yolov2(feature_extractor, num_anchors, num_classes, fine_grain_layer)

    def _construct_yolov2(self, feature_extractor, num_anchors, num_classes, fine_grain_layer):
        """
        Build YOLOv2 Model
        """

        features = feature_extractor if (feature_extractor is not None) else darknet19(freeze_layers=True)
        object_detector = yolov2_detector(features, num_anchors, num_classes, fine_grain_layer=fine_grain_layer)

        YOLOv2 = Model(inputs=[feature_extractor.input], outputs=[object_detector])

        return YOLOv2

    def loss(self):
        raise NotImplemented

    def predict(self):
        raise NotImplemented


def yolov2_detector(feature_extractor, num_anchors, num_classes, fine_grain_layer=43):
    """
    Constructor for Box Regression Model (RPN-ish) for YOLOv2

    :param feature_extractor: 
    :param num_anchors:
    :param num_classes:
    :param fine_grain_layer: default: 43
                layer [(3, 3) 512] of Darknet19 before last max pool
    :return: 
    """
    # Ref: YOLO9000 paper[ "Training for detection" section]

    fine_grained = feature_extractor.layers[fine_grain_layer].output
    fine_grained2= feature_extractor.layers[26].output

    feature_map = feature_extractor.output
    x = _depthwise_conv_block(feature_map, 1024, 1.0, 1, block_id=14)
    x = _depthwise_conv_block(x, 1024, 1.0, 1, block_id=15)

    res_layer = conv_block(fine_grained, 64, (1, 1))
    res_layer2= conv_block(fine_grained2, 64, (1, 1))
    reshaped = Lambda(space_to_depth_x2,
                      space_to_depth_x2_output_shape,
                      name='space_to_depth')(res_layer)

    reshaped2 = Lambda(space_to_depth_x4,
                       space_to_depth_x4_output_shape,
                       name='space_to_depth2')(res_layer2)
    x = concatenate([reshaped2, reshaped, x])

    x = _depthwise_conv_block(x, 1024, 1.0, 1, block_id=16)
    # x = _depthwise_conv_block(x, 1024, 1.0, 1, block_id=17)

    detector = Conv2D(filters=(num_anchors * (num_classes + 5)),
                      kernel_size=(1, 1), kernel_regularizer=l2(5e-4))(x)

    return detector


def space_to_depth_x2(x):
    """Thin wrapper for Tensor flow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.
    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 * input_shape[3]) if input_shape[1] else \
        (input_shape[0], None, None, 4 * input_shape[3])


def space_to_depth_x4(x):
    """Thin wrapper for Tensor flow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=4)


def space_to_depth_x4_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.
    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 4, input_shape[2] // 4, 16 * input_shape[3]) if input_shape[1] else \
        (input_shape[0], None, None, 16 * input_shape[3])
