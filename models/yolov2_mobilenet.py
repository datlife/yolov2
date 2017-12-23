"""
Construct Original YOLOv2 model
"""

from keras.models import Model
from models.net_builder import YOLOv2MetaArch
from models.feature_extractors import mobilenet
from models.detectors.yolov2 import yolov2_detector
from models.preprocessor import yolov2_preprocess_func


def yolov2_mobilenet(inputs,
                     is_training,
                     anchors,
                     num_classes,
                     iou,
                     scores_threshold):

    yolov2 = YOLOv2MetaArch(preprocess_func  = yolov2_preprocess_func,
                            feature_extractor= mobilenet,
                            detector         = yolov2_detector,
                            anchors          = anchors,
                            num_classes      = num_classes)

    prediction = yolov2.predict(inputs)

    if not is_training:
        boxes, classes, scores = yolov2.post_process(prediction,
                                                     iou_threshold  = iou,
                                                     score_threshold= scores_threshold)

        model = Model(inputs=inputs, outputs=[boxes, classes, scores])

    else:
        model = Model(inputs=inputs, outputs=prediction)

    return model
