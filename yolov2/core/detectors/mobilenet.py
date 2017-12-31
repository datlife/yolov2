from keras.layers import concatenate
from ..custom_layers import Reroute
from ..feature_extractors.mobilenet import _depthwise_conv_block


def mobilenet_detector(feature_map,
                       fine_grained_layers):
    """
    MobileNet Detector Implementation
    """
    x = _depthwise_conv_block(feature_map, 1024, 1.0, block_id=14)
    x = _depthwise_conv_block(x, 1024, 1.0, block_id=15)

    # Reroute
    concat_layers = [x]
    for layer in fine_grained_layers:
        connected_layer = _depthwise_conv_block(layer, 64, (1, 1))
        rerouted_layer = Reroute(block_size=2, name='space_to_depth_x2')(connected_layer)
        concat_layers.append(rerouted_layer)

    x = concatenate(concat_layers)
    x = _depthwise_conv_block(x, 1024, (3, 3))

    return x