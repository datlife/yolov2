from keras.layers.merge import concatenate
from keras.layers import MaxPool2D, GlobalAvgPool2D, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, BatchNormalization, Lambda
from keras.models import Model


class YOLOv2(object):
    """
    Yolo V2 Model
    """
    def __init__(self, feature_extractor=None, num_anchors=9, num_classes=31):
        """
        
        :param feature_extractor: Any CNN Classifier. Y
                    YOLOv2 uses Darknet19 (last conv layer as output)
        :param num_anchors: int  
                    - number of anchors   
        :param num_classes: int 
         \          - number of classes in training data
        """
        self.n_anchors = 9
        self.n_classes = 31
        self.model     = self._construct_yolov2(feature_extractor, num_anchors, num_classes)

    def _construct_yolov2(self, feature_extractor, num_anchors, num_classes):
        """
        Build YOLOv2 Model
        """

        features        = feature_extractor if (feature_extractor is not None) else darknet19(freeze_layers=True)
        object_detector = yolov2_detector(features, num_anchors, num_classes, fine_grain_layer=-17)

        yolov2 = Model(inputs=[feature_extractor.input], outputs=[object_detector])

        return yolov2


def darknet19(pretrained_weights=None, freeze_layers=True):
    """
     DarkNet-19 model
     :param img_size          : image size
     :param pretrained_weights: a path to pretrained weights
     :param freeze_layers:      boolean - Freeze layers during training
     
     :return: 
        DarkNet19 model
     """
    image = Input(shape=(None, None, 3))

    x = conv_block(image, 32, (3, 3), padding='same')  # << --- Input layer
    x = MaxPool2D(strides=2)(x)

    x = conv_block(x, 64, (3, 3))
    x = MaxPool2D(strides=2)(x)

    x = conv_block(x, 128, (3, 3))
    x = conv_block(x, 64,  (1, 1))
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
    x = conv_block(x, 1024, (3, 3))

    # x = GlobalAvgPool2D(1000, (1, 1), )(x)  - disabled because we only need the last conv layer to extract features
    feature_extractor = Model(image, x)

    if pretrained_weights is not None:  # Load pre-trained weights from DarkNet19
        feature_extractor.load_weights(pretrained_weights, by_name=True)
        print("Pre-trained weights have been loaded into model")

    if freeze_layers:
        for layer in feature_extractor.layers:
            layer.trainable = False

    return feature_extractor


def yolov2_detector(feature_extractor, num_anchors, num_classes, fine_grain_layer=-17):
    """
    Constructor for Box Regression Model for YOLOv2
    
    :param feature_extractor: 
    :param num_anchors:
    :param num_classes:
    :param fine_grain_layer: default: -17
                layer [(3, 3) 512] of Darknet19 before last max pool
    :return: 
    """
    # Ref: YOLO9000 paper[ "Training for detection" section]

    fine_grained = feature_extractor.layers[fine_grain_layer].output
    feature_map  = feature_extractor.output

    detector     = conv_block(feature_map, 1024, (3, 3))
    detector     = conv_block(detector,    1024, (3, 3))

    i            = conv_block(fine_grained, 64,  (1, 1))      # concatenate pass-through layer with current conv layer
    reshaped     = Lambda(func_reshape, get_output_shape)(i)
    detector     = concatenate([reshaped, detector])

    detector     = conv_block(detector,    1024, (3, 3))
    detector     = Conv2D(filters=(num_anchors * (num_classes + 5)), kernel_size=(1, 1), name='yolov2')(detector)

    return detector


def func_reshape(x):
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def get_output_shape(input_shape):
    if input_shape[1]:
        return input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 * input_shape[3]
    else:
        return input_shape[0], None, None, 4 * input_shape[3]


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
    x = LeakyReLU()(x)
    return x


# Test case
if __name__ == "__main__":

    darknet19 = darknet19(freeze_layers=True)
    yolov2    = YOLOv2(feature_extractor=darknet19, num_anchors=5, num_classes=31)
    yolov2.model.summary()
    model = yolov2.model
    import keras.backend as K
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    print(K.shape(model.output))
