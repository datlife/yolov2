"""
Construct Original YOLOv2 model
"""

from .net_builder import YOLOv2MetaArch
from .feature_extractors import mobilenet
from .detectors.yolov2 import yolov2_detector


def yolov2_mobilenet(inputs,
                     is_training,
                     anchors,
                     num_classes,
                     iou,
                     scores_threshold):

    yolov2 = YOLOv2MetaArch(feature_extractor= mobilenet,
                            detector         = yolov2_detector,
                            anchors          = anchors,
                            num_classes      = num_classes)

    predictions = yolov2.predict(inputs)

    if not is_training:
        boxes, classes, scores = yolov2.post_process(predictions,
                                                     iou_threshold  = iou,
                                                     score_threshold= scores_threshold)
        return boxes, classes, scores

    else:
        return predictions
