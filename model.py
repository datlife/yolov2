from yolov2.model import YOLOv2
from yolov2.core.detectors import yolov2_detector
from yolov2.core.feature_extractors import darknet19


def yolov2(is_training, config):
    model = YOLOv2(is_training=is_training,
                   feature_extractor=darknet19,
                   detector=yolov2_detector,
                   config_dict=config)
    return model
