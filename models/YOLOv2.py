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
from keras.models import Model
from keras.regularizers import l2
from cfg import *


class YOLOv2(object):
    """
    YOLOv2 Meta-Architecture
    """
    def __init__(self,
                 feature_extractor=None,
                 num_anchors=N_ANCHORS,
                 num_classes=N_CLASSES,
                 fine_grain_layer=[],
                 dropout=None):

        self.model = self._construct_yolov2(feature_extractor, num_anchors, num_classes, fine_grain_layer)

    def _construct_yolov2(self, feature_extractor, num_anchors, num_classes, fine_grain_layer):
        """
        Build YOLOv2 Model
        Input :  feature map from feature extractor
        Ouput :  prediction
        """
        fine_grained = feature_extractor.get_layer(name=fine_grain_layer[0]).output
        feature_map = feature_extractor.output
        x = conv_block(feature_map, 1024, (3, 3))
        x = conv_block(x, 1024, (3, 3))

        res_layer = conv_block(fine_grained, 64, (1, 1))
        reshaped = Lambda(space_to_depth_x2, space_to_depth_x2_output_shape, name='space_to_depth')(res_layer)

        x = concatenate([reshaped, x])
        x = conv_block(x, 1024, (3, 3))

        detector = Conv2D(filters=(num_anchors * (num_classes + 5)), name='yolov2',
                          kernel_size=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-4))(x)
        YOLOv2 = Model(inputs=[feature_extractor.input], outputs=detector)
        return YOLOv2