from keras.layers import concatenate
from ..custom_layers import Reroute
from ..feature_extractors.darknet19 import conv_block


def yolov2_detector(feature_map,
                    fine_grained_layers):
    """
        Original YOLOv2 Implementation
    """
    # @TODO : create a loop for fine_grained layers
    layer = fine_grained_layers[0]

    x = conv_block(feature_map, 1024, (3, 3))
    x = conv_block(x, 1024, (3, 3))
    x2 = x

    connected_layer = conv_block(layer, 64, (1, 1))
    rerouted_layer  = Reroute(block_size=2,
                              name='space_to_depth_x2')(connected_layer)

    x = concatenate([rerouted_layer, x2])
    x = conv_block(x, 1024, (3, 3))

    return x
