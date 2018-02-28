"""
YOLOv2 Meta-Architecture

        images --> feature extractor --> feature map --> detector --> output feature map

In this file, we define three different detector:
   * Original YOLOv2 Detector
   * MobileNet-type detector

-----------------------
Example usage: This code will define pretrained YOLOv2 on COCO Dataset (80 classes)
"""
import tensorflow as tf

from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Lambda
from .custom_layers import PostProcessor, OutputInterpreter
# from tensorflow.python.keras.regularizers import l2

K = tf.keras.backend


class YOLOv2MetaArch(object):
  def __init__(self,
               feature_extractor,
               detector,
               anchors,
               num_classes,
               init_weights=None):
    """
    YOLOv2 meta architecture, it consists of:
        * Preprocessor      - a custom Keras layer that pre-process inputs
        * Feature Extractor - a FeatureExtractor object
        * Detector          - a Detector Object
    :param feature_extractor:
    :param detector:
    :param anchors:
    :param num_classes:
    """

    self.anchors = anchors
    self.num_classes = num_classes
    self.feature_extractor = feature_extractor
    self.detector = detector
    self.init_weights =init_weights

  def predict(self, resized_inputs):
    with tf.name_scope('YOLOv2'):
      feature_map, pass_through_layers = self.feature_extractor(resized_inputs)

      x = self.detector(feature_map, pass_through_layers)
      if self.init_weights:
        model = tf.keras.models.Model(resized_inputs, x).load_weights(self.init_weights)
        x = model.outputs

      x = Conv2D(len(self.anchors) * (self.num_classes + 5), (1, 1),
                 name='OutputFeatures')(x)

      x = OutputInterpreter(anchors=self.anchors,
                            num_classes=self.num_classes,
                            name='Predictions')(x)
      return x

  @staticmethod
  def post_process(predictions, iou_threshold, score_threshold, max_boxes=100):
    """
    Preform non-max suppression to calculate outputs:
    Using during evaluation/inference
    Args:
        out feature map from network
    Output:
       Bounding Boxes - Classes - Probabilities
    """

    outputs = PostProcessor(score_threshold=score_threshold,
                            iou_threshold=iou_threshold,
                            max_boxes=max_boxes,
                            name="NonMaxSuppression")(predictions)

    boxes = Lambda(lambda x: x[..., :4], name="boxes")(outputs)
    scores = Lambda(lambda x: x[..., 4], name="scores")(outputs)
    classes = Lambda(lambda x: K.cast(x[..., 5], tf.float32), name="classes")(outputs)
    return boxes, classes, scores
