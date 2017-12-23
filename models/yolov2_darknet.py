"""
Construct Original YOLOv2 model
"""

from models.net_builder import YOLOv2MetaArch
from models.custom_layers import ImageResizer

from models.preprocessor import yolov2_preprocess_func
from models.feature_extractors import darknet19
from models.detectors.yolov2 import yolov2_detector


def yolov2_darknet(inputs,
                   img_size,
                   is_training,
                   anchors,
                   num_classes,
                   iou,
                   scores_threshold):

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

