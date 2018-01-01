"""
Build YOLOv2 Model

The idea is that YOLOv2 consists of feature extractor and detector.
By using YOLOv2MetaArch, one can swap different types o feature extractor (DarkNet19, MobileNet, NASNet, DenseNet)
and different types of detector, too.

In this file, we construct a standard YOLOv2 using Darknet19 as feature extractor.
"""
from keras.layers import Input
from keras.models import Model
from yolov2.core.net_builder import YOLOv2MetaArch

from yolov2.core.feature_extractors import darknet19
from yolov2.core.detectors.yolov2 import yolov2_detector
from yolov2.core.custom_layers import ImageResizer


def yolov2_darknet19(img_size,
                     is_training,
                     anchors,
                     num_classes,
                     iou=0.5,
                     scores_threshold=0.0,
                     max_boxes = 100):
    """Definition of YOLOv2 using DarkNet19 as feature extractor

    :param img_size:    - an int - default image size that let ImageResizer to know how to resize the image
    :param is_training: - a boolean
    :param anchors:     - a float numpy array - list of anchors
    :param num_classes: - an int - number of classes in the dataset
    :param iou:         - a float - Intersection over Union value (only used when is_training = False)
    :param scores_threshold: - a float - Minimum accuracy value (only used when is_training = False)
    :param max_boxes    - an int - maximum of boxes used in Post Process (non-max-suppression)
    :return: the outputs of model
    """
    inputs = Input(shape=(None, None, 3), name='image_input')

    # Construct Keras model, this function return a build blocks
    # of YOLOv2 model
    yolov2 = YOLOv2MetaArch(feature_extractor= darknet19,
                            detector         = yolov2_detector,
                            anchors          = anchors,
                            num_classes      = num_classes)

    resized_inputs = ImageResizer(img_size, name="ImageResizer")(inputs)
    outputs = yolov2.predict(resized_inputs)

    if is_training:
        return Model(inputs=inputs, outputs=outputs)
    else:
        outputs = yolov2.post_process(outputs, iou, scores_threshold, max_boxes)
        return Model(inputs=inputs, outputs=outputs)

