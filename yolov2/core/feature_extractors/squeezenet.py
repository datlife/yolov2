import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Activation
from keras.layers import concatenate
from keras.layers import GlobalAvgPool2D


def squeezenet_preprocces_func(image):
    image = image[..., ::-1]
    image = image.astype('float')
    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68
    return image


def squeezenet(inputs, num_classes=1000, include_top=False):

    with tf.name_scope('SqueezeNet'):
        fine_grained_layers = []

        # define the model of SqueezeNet
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inputs)
        x = Activation('relu', name='relu_conv1')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)
        fine_grained_layers.append(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)

        if include_top:
            x = Conv2D(num_classes, (1, 1), padding='valid', name='conv10')(x)
            x = Activation('relu', name='relu_conv10')(x)
            x = GlobalAvgPool2D()(x)
            x = Activation('softmax', name='loss')(x)

        return x, fine_grained_layers


def fire_module(x, fire_id, squeeze=16, expand=64):
    # define some auxiliary variables and the fire module
    sq1x1  = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu   = "relu_"

    with tf.name_scope('FireModule_%s' % str(fire_id)):
        s_id = 'fire' + str(fire_id) + '/'

        x     = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
        x     = Activation('relu', name=s_id + relu + sq1x1)(x)

        left  = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
        left  = Activation('relu', name=s_id + relu + exp1x1)(left)

        right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
        right = Activation('relu', name=s_id + relu + exp3x3)(right)

        x = concatenate([left, right], axis=3, name=s_id + 'concat')

        return x
