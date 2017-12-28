"""
Construct Original YOLOv2 model
"""
from keras.layers import Input
from keras.models import Model
from .net_builder import YOLOv2MetaArch

from .feature_extractors import darknet19
from .custom_layers import ImageResizer
from .detectors.yolov2 import yolov2_detector


def yolov2_darknet(img_size,
                   is_training,
                   anchors,
                   num_classes,
                   iou,
                   scores_threshold):
    """Definition of YOLOv2 using DarkNet19 as feature extractor

    :param img_size:    - a int - default image size that let ImageResizer to know how to resize the image
    :param is_training: - a boolean
    :param anchors:     - a numpy of float array - list of anchors
    :param num_classes: - a int - number of classes in the dataset
    :param iou:         - a float - Intersection over Union value (only used when is_training = False)
    :param scores_threshold: - a float - Minimum accuracy value (only used when is_training = False)

    :return: the outputs of model
    """
    inputs = Input(shape=(None, None, 3), name='image_input')

    yolov2 = YOLOv2MetaArch(feature_extractor= darknet19,
                            detector         = yolov2_detector,
                            anchors          = anchors,
                            num_classes      = num_classes)

    resized_inputs = ImageResizer(img_size, name="ImageResizer")(inputs)
    outputs    = yolov2.predict(resized_inputs)

    if is_training:
        return Model(inputs=inputs, outputs=outputs)

    else:
        outputs = yolov2.post_process(outputs,
                                      iou_threshold  = iou,
                                      score_threshold= scores_threshold)
        return Model(inputs=inputs, outputs=outputs)

