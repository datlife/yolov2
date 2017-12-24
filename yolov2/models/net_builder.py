"""
YOLOv2 Meta-Architecture

        images --> feature extractor --> feature map --> detector --> output feature map

In this file, we define three different detector:
   * Original YOLOv2 Detector
   * MobileNet-type detector
   * DenseNet-type detector

-----------------------
Example usage: This code will define pretrained YOLOv2 on COCO Dataset (80 classes)
"""
import tensorflow as tf
import keras.backend as K

from keras.layers import Conv2D
from keras.layers import Lambda
from .custom_layers import Preprocessor, PostProcessor


class YOLOv2MetaArch(object):
    def __init__(self,
                 preprocess_func,
                 feature_extractor,
                 detector,
                 anchors,
                 num_classes):

        self.anchors             = anchors
        self.num_classes         = num_classes
        self.preprocess_func     = preprocess_func

        self.feature_extractor   = feature_extractor
        self.detector            = detector

    def predict(self, resized_inputs):

        preprocessed_inputs  = Preprocessor(self.preprocess_func,
                                            name='preprocessor')(resized_inputs)
        # ##################
        # Feature Extractor
        # ##################
        feature_map, fine_grained_layers = self.feature_extractor(preprocessed_inputs)

        # ################
        # Object Detector
        # ################
        outputs = self.detector(feature_map, fine_grained_layers)

        predictions = Conv2D(len(self.anchors) * (self.num_classes + 5),
                             (1, 1), name='yolov2')(outputs)

        return predictions

    def post_process(self, prediction, iou_threshold=0.6, score_threshold=0.6):
        """
        Preform non-max suppression to calculate outputs:
        Using during evaluation/interference
        Args:
            out feature map from network
        Output:
           Bounding Boxes - Classes - Probabilities
        """
        outputs = PostProcessor(score_threshold,
                                iou_threshold,
                                interpret_prediction,  # this is a function
                                self.anchors,
                                self.num_classes,
                                name="non_max_suppression")(prediction)
        boxes   = Lambda(lambda x: x[..., :4], name="boxes")(outputs)
        scores  = Lambda(lambda x: x[..., 4],  name="scores")(outputs)
        classes = Lambda(lambda x: K.cast(x[..., 5], tf.float32),  name="classes")(outputs)

        return boxes, classes, scores

# ##############
# HELPER METHODS
# ##############


def interpret_prediction(prediction, anchors, num_classes):
    N_CLASSES = num_classes
    N_ANCHORS = len(anchors)
    ANCHORS = anchors

    pred_shape = tf.shape(prediction)
    GRID_H, GRID_W = pred_shape[1], pred_shape[2]

    prediction = K.reshape(prediction, [-1, pred_shape[1], pred_shape[2], N_ANCHORS, N_CLASSES + 5])

    # Create off set map
    cx = tf.cast((K.arange(0, stop=GRID_W)), dtype=tf.float32)
    cx = K.tile(cx, [GRID_H])
    cx = K.reshape(cx, [-1, GRID_H, GRID_W, 1])

    cy = K.cast((K.arange(0, stop=GRID_H)), dtype=tf.float32)
    cy = K.reshape(cy, [-1, 1])
    cy = K.tile(cy, [1, GRID_W])
    cy = K.reshape(cy, [-1])
    cy = K.reshape(cy, [-1, GRID_H, GRID_W, 1])

    c_xy = tf.stack([cx, cy], -1)
    c_xy = tf.to_float(c_xy)

    anchors_tensor = tf.to_float(K.reshape(ANCHORS, [1, 1, 1, N_ANCHORS, 2]))
    netout_size = tf.to_float(K.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]))

    box_xy = K.sigmoid(prediction[..., :2])
    box_wh = K.exp(prediction[..., 2:4])
    box_confidence = K.sigmoid(prediction[..., 4:5])
    box_class_probs = K.softmax(prediction[..., 5:])

    # Shift center points to its grid cell accordingly (Ref: YOLO-9000 loss function)
    box_xy = (box_xy + c_xy) / netout_size
    box_wh = (box_wh * anchors_tensor) / netout_size

    return box_xy, box_wh, box_confidence, box_class_probs

# import numpy as np
# import tensorflow as tf
# import keras.backend as K
#
# from zoo.darknet19 import conv_block
# from zoo.mobilenet import _depthwise_conv_block, relu6
# 
# from keras.models import Model
# from keras.regularizers import l2
# from keras.layers.merge import concatenate
# from keras.layers import Lambda, Conv2D, BatchNormalization, Activation
#
# from softmaxtree.Tree import SoftMaxTree
#
# FINE_GRAINED_LAYERS = {'yolov2': 'leaky_re_lu_13',
#                        'mobilenet': 'conv_pw_11_relu',
#                        'densenet': 'leaky_re_lu_136'}

# x = conv_block(x, 32, (3, 3))  # << --- Start feature extraction
# x = MaxPool2D(strides=2)(x)
#
# x = conv_block(x, 64, (3, 3))
# x = MaxPool2D(strides=2)(x)
#
# x = conv_block(x, 128, (3, 3))
# x = conv_block(x, 64, (1, 1))
# x = conv_block(x, 128, (3, 3))
# x = MaxPool2D(strides=2)(x)
#
# x = conv_block(x, 256, (3, 3))
# x = conv_block(x, 128, (1, 1))
# x = conv_block(x, 256, (3, 3))
# x = MaxPool2D(strides=2)(x)
#
# x = conv_block(x, 512, (3, 3))
# x = conv_block(x, 256, (1, 1))
# x = conv_block(x, 512, (3, 3))
# x = conv_block(x, 256, (1, 1))
# x = conv_block(x, 512, (3, 3))
# fine_grained_layer = x
# x = MaxPool2D(strides=2)(x)
#
# x = conv_block(x, 1024, (3, 3))
# x = conv_block(x, 512, (1, 1))
# x = conv_block(x, 1024, (3, 3))
# x = conv_block(x, 512, (1, 1))
# feature_map = conv_block(x, 1024, (3, 3))  # ---> feature extraction ends here


# LOSS

"""
YOLOv2 Loss Function Implementation

Input: out feature map from network

Output:
   A scalar - loss value for back propagation

------------------

Loss of YOLOv2 Implementation. Few Notes
    * What we get from the CNN is a feature map (imagine as a 3-D box)
    * Each cell in a feature map is a vector size N_ANCHORS* (5 + N_CLASSES)  as:

        ------------- ANCHOR 1---------------  -------- ANCHORS 2 -------------  ...... ------------ANCHOR N -----------
        [tx, ty, tw, th, to , label_vector..],[tx1, ty1, tw1, th1, label_vector]......[tx_n, ty_n, tw_n, th_m, label...]
        ----------------------------------------------------------------------------------------------------------------
                                                 One cell in a feature map


            * tx, ty : predicts of relative center of bounding box to its current cell. Therefore, true center points of
                       a prediction would be :
                            xc = sigmoid(tx) + cx
                            yc = sigmoid(ty) + cy
            * tw, th: predicts the scaling value for true width and height of the bounding box based on the anchor as:
                            w   = exp(tw) * px
                            h   = exp(th) * py
            * to :     objectiveness of the cell : the probability of having an object in the cell
            * label:   classification vector to calculate soft-max
"""
# import re
# import numpy as np
# import tensorflow as tf
# import keras.backend as K
# from cfg import *
# from softmaxtree.Tree import SoftMaxTree
#
#
# def custom_loss(y_true, y_pred):
#     """
#     Loss Function of YOLOv2
#     :param y_true: a Tensor  [batch_size, GRID_H, GRID_W, N_ANCHORS*(N_CLASSES + 5)]
#     :param y_pred: a Tensor  [batch_size, GRID_H, GRID_H, N_ANCHOR*(N_CLASSES + 5)]
#     :return: a scalar
#             loss value
#     """
#     if ENABLE_HIERARCHICAL_TREE is True:
#         SOFTMAX_TREE = SoftMaxTree(tree_file=HIERARCHICAL_TREE_PATH)
#
#     # Config Anchors
#     anchors = []
#     with open(ANCHORS, 'r') as f:
#       data = f.read().splitlines()
#       for line in data:
#         numbers = re.findall('\d+.\d+', line)
#         anchors.append((float(numbers[0]), float(numbers[1])))
#     anchors = np.array(anchors)
#
#     pred_shape = K.shape(y_pred)[1:3]
#     gt_shape = K.shape(y_true)  # shape of ground truth value
#     GRID_H = tf.cast(pred_shape[0], tf.int32)  # shape of output feature map
#     GRID_W = tf.cast(pred_shape[1], tf.int32)
#
#     output_size = tf.cast(tf.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]), tf.float32)
#     y_pred = tf.reshape(y_pred, [-1, pred_shape[0], pred_shape[1], N_ANCHORS, N_CLASSES + 5])
#     y_true = tf.reshape(y_true, [-1, gt_shape[1], gt_shape[2], N_ANCHORS, N_CLASSES + 5])
#
#     # Grid Map to calculate offset
#     c_xy = _create_offset_map(K.shape(y_pred))
#
#     # Scale anchors to correct aspect ratio
#     pred_box_xy   = (tf.sigmoid(y_pred[:, :, :, :, :2]) + c_xy) / output_size
#     pred_box_wh   = tf.exp(y_pred[:, :, :, :, 2:4]) * np.reshape(anchors, [1, 1, 1, N_ANCHORS, 2]) / output_size
#     pred_box_wh   = tf.sqrt(pred_box_wh)
#     pred_box_conf = tf.sigmoid(y_pred[:, :, :, :, 4:5])
#     pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])
#
#     # Adjust ground truth
#     true_box_xy = y_true[:, :, :, :, 0:2]
#     true_box_wh = tf.sqrt(y_true[:, :, :, :, 2:4])
#
#     # adjust confidence
#     pred_tem_wh   = tf.pow(pred_box_wh, 2) * output_size
#     pred_box_ul   = pred_box_xy - 0.5 * pred_tem_wh
#     pred_box_bd   = pred_box_xy + 0.5 * pred_tem_wh
#     pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
#
#     true_tem_wh   = tf.pow(true_box_wh, 2) * output_size
#     true_box_ul   = true_box_xy - 0.5 * true_tem_wh
#     true_box_bd   = true_box_xy + 0.5 * true_tem_wh
#     true_box_area = true_tem_wh[:, :, :, :, 0] * true_tem_wh[:, :, :, :, 1]
#
#     intersect_ul   = tf.maximum(pred_box_ul, true_box_ul)
#     intersect_br   = tf.minimum(pred_box_bd, true_box_bd)
#     intersect_wh   = tf.maximum(intersect_br - intersect_ul, 0.0)
#     intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#
#     iou           = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
#     best_box      = tf.equal(iou, tf.reduce_max(iou, [3], True))
#     best_box      = tf.to_float(best_box)
#     true_box_conf = tf.expand_dims(best_box * y_true[:, :, :, :, 4], -1)
#     true_box_prob = y_true[:, :, :, :, 5:]
#
#     # Localization Loss
#     weight_coor = 5.0 * tf.concat(4 * [true_box_conf], 4)
#     true_boxes  = tf.concat([true_box_xy, true_box_wh], 4)
#     pred_boxes  = tf.concat([pred_box_xy, pred_box_wh], 4)
#     loc_loss    = tf.pow(true_boxes - pred_boxes, 2) * weight_coor
#     loc_loss    = tf.reshape(loc_loss, [-1, tf.cast(GRID_W * GRID_H, tf.int32) * N_ANCHORS * 4])
#     loc_loss    = tf.reduce_mean(tf.reduce_sum(loc_loss, 1))
#
#     # NOTE: YOLOv2 does not use cross-entropy loss.
#     if ENABLE_HIERARCHICAL_TREE is False:
#         # Object Confidence Loss
#         weight_conf = 0.5 * (1. - true_box_conf) + 5.0 * true_box_conf
#         obj_conf_loss = tf.pow(true_box_conf - pred_box_conf, 2) * weight_conf
#         obj_conf_loss = tf.reshape(obj_conf_loss, [-1, tf.cast(GRID_W * GRID_H, tf.int32) * N_ANCHORS])
#         obj_conf_loss = tf.reduce_mean(tf.reduce_sum(obj_conf_loss, 1))
#
#         # Category Loss
#         weight_prob = 1.0 * tf.concat(N_CLASSES * [true_box_conf], 4)
#         category_loss = tf.pow(true_box_prob - pred_box_prob, 2) * weight_prob
#         category_loss = tf.reshape(category_loss, [-1, tf.cast(GRID_W * GRID_H, tf.int32) * N_ANCHORS * N_CLASSES])
#         category_loss = tf.reduce_mean(tf.reduce_sum(category_loss, 1))
#
#         loss = 0.5 * (loc_loss + obj_conf_loss + category_loss)
#
#     else:
#         category_loss = SOFTMAX_TREE.calculate_softmax(idx=0, logits=y_pred[..., 4:], labels=y_true[..., 4:],
#                                                        pred_obj_conf=pred_box_conf, true_obj_conf=true_box_conf)
#         # category_loss = tf.Print(category_loss, [category_loss])
#         loss = 0.5 * (loc_loss + category_loss)
#
#     return loss
#
#
# def _create_offset_map(output_shape):
#     """
#     In Yolo9000 paper, Grid map to calculate offsets for each cell in the output feature map
#     """
#     GRID_H = tf.cast(output_shape[1], tf.int32)  # shape of output feature map
#     GRID_W = tf.cast(output_shape[2], tf.int32)
#
#     cx = tf.cast((K.arange(0, stop=GRID_W)), dtype=tf.float32)
#     cx = K.tile(cx, [GRID_H])
#     cx = K.reshape(cx, [-1, GRID_H, GRID_W, 1])
#
#     cy = K.cast((K.arange(0, stop=GRID_H)), dtype=tf.float32)
#     cy = K.reshape(cy, [-1, 1])
#     cy = K.tile(cy, [1, GRID_W])
#     cy = K.reshape(cy, [-1])
#     cy = K.reshape(cy, [-1, GRID_H, GRID_W, 1])
#
#     c_xy = tf.stack([cx, cy], -1)
#     c_xy = K.cast(c_xy, tf.float32)
#
#     return c_xy