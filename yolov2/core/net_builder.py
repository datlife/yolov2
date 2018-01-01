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
import keras.backend as K

from keras.layers import Conv2D
from keras.layers import Lambda
from .custom_layers import PostProcessor


class YOLOv2MetaArch(object):
    def __init__(self,
                 feature_extractor,
                 detector,
                 anchors,
                 num_classes):
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

        self.anchors             = anchors
        self.num_classes         = num_classes
        self.feature_extractor   = feature_extractor
        self.detector            = detector

    def predict(self, resized_inputs):

        with tf.name_scope('YOLOv2'):
            # Feature Extractor
            feature_map, pass_through_layers = self.feature_extractor(resized_inputs)

            with tf.name_scope("Detector"):
                x       = self.detector(feature_map, pass_through_layers)
                x       = Conv2D(len(self.anchors) * (self.num_classes + 5), (1, 1),
                                 name='output_features')(x)

                predictions = Lambda(lambda x: self.interpret_yolov2(x),
                                     name='predictions')(x)
            return predictions

    def post_process(self, predictions, iou_threshold, score_threshold, max_boxes=100):
        """
        Preform non-max suppression to calculate outputs:
        Using during evaluation/interference
        Args:
            out feature map from network
        Output:
           Bounding Boxes - Classes - Probabilities
        """
        outputs = PostProcessor(score_threshold= score_threshold,
                                iou_threshold  = iou_threshold,
                                max_boxes      = max_boxes,
                                name="NonMaxSuppression")(predictions)

        boxes   = Lambda(lambda x: x[..., :4], name="boxes")(outputs)
        scores  = Lambda(lambda x: x[..., 4],  name="scores")(outputs)
        classes = Lambda(lambda x: K.cast(x[..., 5], tf.float32),  name="classes")(outputs)
        return boxes, classes, scores

    def interpret_yolov2(self, predictions):
        shape = tf.shape(predictions)
        height, width = shape[1], shape[2]

        #  @TODO: waiting for Tf.repeat() in coming TF version
        # ##################
        #  Create offset map
        # ##################
        cx   = tf.reshape(tf.tile(tf.range(width), [height]), [-1, height, width, 1])
        cy   = tf.tile(tf.expand_dims(tf.range(height), -1), [1, width])
        cy   = tf.reshape(cy, [-1, height, width, 1])
        c_xy = tf.to_float(tf.stack([cx, cy], -1))

        anchors_tensor = tf.to_float(K.reshape(self.anchors, [1, 1, 1, len(self.anchors), 2]))
        output_size    = tf.to_float(K.reshape([width, height], [1, 1, 1, 1, 2]))

        outputs = K.reshape(predictions, [-1, height, width, len(self.anchors), self.num_classes + 5])

        # ##################
        # Interpret outputs
        # ##################
        #  (Ref: YOLO-9000 paper)
        box_xy          = K.sigmoid(outputs[..., :2]) + c_xy
        box_wh          = K.exp(outputs[..., 2:4]) * anchors_tensor
        box_confidence  = K.sigmoid(outputs[..., 4:5])
        box_class_probs = K.softmax(outputs[..., 5:])

        # Convert coordinates to relative coordinates (percentage)
        box_xy = box_xy / output_size
        box_wh = box_wh / output_size

        # Calculate corner points of bounding boxes
        box_mins  = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)

        # Y1, X1, Y2, X2
        boxes = K.concatenate([box_mins[..., 1:2], box_mins[..., 0:1],     # Y1 X1
                               box_maxes[..., 1:2], box_maxes[..., 0:1]])  # Y2 X2

        outputs = K.concatenate([boxes, box_confidence, box_class_probs], axis=-1)

        return outputs

    def compute_loss(self, predictions):
        """
        Keras does not support loss function during graph construction.
        A loss function can only be pass during model.compile(loss=loss_func)
        """
        raise NotImplemented
