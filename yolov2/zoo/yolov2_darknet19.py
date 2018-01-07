"""
Build YOLOv2 Model

The idea is that YOLOv2 consists of feature extractor and detector.
By using YOLOv2MetaArch, one can swap different types o feature extractor (DarkNet19, MobileNet, NASNet, DenseNet)
and different types of detector, too.

In this file, we construct a standard YOLOv2 using Darknet19 as feature extractor.
"""
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model

from yolov2.core.loss import YOLOV2Loss
from yolov2.core.net_builder import YOLOv2MetaArch

from yolov2.core.custom_layers import ImageResizer
from yolov2.core.feature_extractors import darknet19
from yolov2.core.detectors.yolov2 import yolov2_detector


def yolov2_darknet19(is_training,
                     anchors,
                     num_classes,
                     config):
    """Definition of YOLOv2 using DarkNet19 as feature extractor

    :return:
        the outputs of model
    """
    # Construct Keras model, this function return a build blocks
    # of YOLOv2 model
    # @TODO: Fix ImageResizer placeholder
    resized_inputs = Input(shape=(None, None, 3), name='input_images')

    yolov2 = YOLOv2MetaArch(feature_extractor= darknet19,
                            detector         = yolov2_detector,
                            anchors          = anchors,
                            num_classes      = num_classes)

    outputs = yolov2.predict(resized_inputs)

    if is_training:
        learning_rate = config['training_params']['learning_rate']

        objective_function = YOLOV2Loss(anchors, num_classes)
        model = Model(inputs=resized_inputs, outputs=outputs)
        model.compile(optimizer=Adam(lr=learning_rate),
                      loss=objective_function.compute_loss)
    else:
        deploy_params = config['deploy_params']
        outputs = yolov2.post_process(outputs,
                                      deploy_params['iou_threshold'],
                                      deploy_params['score_threshold'],
                                      deploy_params['maximum_boxes'])
        model   = Model(inputs=resized_inputs, outputs=outputs)

    model.load_weights(config['model']['weight_file'])
    print("Weight file has been loaded in to model")

    return model

