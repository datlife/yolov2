"""
DarKNet 19 Architecture
"""
from model.net_builder import conv_block
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.models import Model


def darknet19(input_size=None, pretrained_weights=None, freeze_layers=True):
    """
     DarkNet-19 model
     :param img_size          : image size
     :param pretrained_weights: a path to pretrained weights
     :param freeze_layers:      boolean - Freeze layers during training

     :return: 
        DarkNet19 model
     """
    if input_size is None:
        image = Input(shape=(None, None, 3))
    else:
        image = Input(shape=input_size)

    x = conv_block(image, 32, (3, 3), padding='same')  # << --- Input layer
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
