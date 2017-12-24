"""
Construct Original YOLOv2 model
"""

from models.net_builder import YOLOv2MetaArch
from models.custom_layers import ImageResizer

from models.feature_extractors import darknet19
from models.detectors.yolov2 import yolov2_detector
from models.preprocessor import yolov2_preprocess_func


def yolov2_darknet(inputs,
                   img_size,
                   is_training,
                   anchors,
                   num_classes,
                   iou,
                   scores_threshold):
    """Definition of YOLOv2 using DarkNet19 as feature extractor

    :param inputs:      - a Input keras layer - a placeholder of batch of imageas
    :param img_size:    - a int - default image size that let ImageResizer to know how to resize the image
    :param is_training: - a boolean
    :param anchors:     - a numpy of float array - list of anchors
    :param num_classes: - a int - number of classes in the dataset
    :param iou:         - a float - Intersection over Union value (only used when is_training = False)
    :param scores_threshold: - a float - Minimum accuracy value (only used when is_training = False)

    :return: the outputs of model
    """

    yolov2 = YOLOv2MetaArch(preprocess_func  = yolov2_preprocess_func,
                            feature_extractor= darknet19,
                            detector         = yolov2_detector,
                            anchors          = anchors,
                            num_classes      = num_classes)

    resized_inputs = ImageResizer(img_size, name="ImageResizer")(inputs)
    predictions    = yolov2.predict(resized_inputs)

    if is_training:
        return predictions

    else:
        boxes, classes, scores = yolov2.post_process(predictions,
                                                     iou_threshold  = iou,
                                                     score_threshold= scores_threshold)
        return boxes, classes, scores

