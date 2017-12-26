"""
Loss (Objective) Function definition for You Only Look Once version 2
"""
import numpy as np
import tensorflow as tf
import keras.backend as K


# def yolov2_loss(anchors, num_classes):
#
#     def compute_localization_loss(gt_boxes, pred_boxes):
#         return  0.0
#
#     def compute_classification_loss(gt_classes, pred_classes)
#         return 0.0
#
#     def compute_loss(y_true, y_pred):
#
#         gt_boxes   = []
#         pred_boxes = []
#
#         gt_classes   = []
#         pred_classes = []
#
#         localization_loss   = compute_localization_loss(gt_boxes, pred_boxes)
#         classification_loss = compute_classification_loss(gt_classes, pred_classes)
#
#         total_loss = 1.0 * localization_loss + \
#                      1.0 * classification_loss
#         return total_loss
#
#     return compute_loss

#
# def iou(boxeslist1, boxeslist2, scope=None):
#     """
#       Args:
#         boxeslist1: BoxList holding N boxes
#         boxeslist2: BoxList holding M boxes
#         scope
#
#       Returns:
#         a tensor with shape [N, M] representing pairwise iou scores.
#       """
#     with tf.name_scope(scope, 'IOU'):
#         intersections = intersection(boxeslist1, boxeslist2)
#         areas1 = area(boxeslist1)
#         areas2 = area(boxeslist2)
#
#         unions = (tf.expand_dims(areas1, 1) +
#                   tf.expand_dims(areas2, 0) - intersections)
#
#     return tf.where(tf.equal(intersections, 0.0),
#                     tf.zeros_like(intersections),
#                     tf.truediv(intersections, unions))
# #
#
# def area(boxes, scope=None):
#     """Computes area of boxes.
#
#     Args:
#       boxes: BoxList holding N boxes
#       scope: name scope.
#
#     Returns:
#       a tensor with shape [N] representing box areas.
#     """
#     with tf.name_scope(scope, 'Area'):
#         y_min, x_min, y_max, x_max = tf.split(value=boxes.get(), num_or_size_splits=4, axis=1)
#
#         return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])
#
#
# def intersection(boxlist1, boxlist2, scope=None):
#     """Compute pairwise intersection areas between boxes.
#
#     Args:
#     boxlist1: BoxList holding N boxes
#     boxlist2: BoxList holding M boxes
#     scope: name scope.
#
#     Returns:
#     a tensor with shape [N, M] representing pairwise intersections
#     """
#     with tf.name_scope(scope, 'Intersection'):
#         y_min1, x_min1, y_max1, x_max1 = tf.split(value=boxlist1.get(), num_or_size_splits=4, axis=1)
#         y_min2, x_min2, y_max2, x_max2 = tf.split(value=boxlist2.get(), num_or_size_splits=4, axis=1)
#
#         all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
#         all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
#
#         intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
#         all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
#         all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
#
#         intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
#
#     return intersect_heights * intersect_widths

# def loss_func(y_true, y_pred):
#     """c
#     YOLOv2 Loss Function Implementation
#     Output:
#        A scalar - loss value for back propagation
#     """
#     N_ANCHORS = len(self.anchors)
#     N_CLASSES = self.num_classes
#
#     pred_shape = K.shape(y_pred)[1:3]
#     GRID_H     = tf.cast(pred_shape[0], tf.int32)  # shape of output feature map
#     GRID_W     = tf.cast(pred_shape[1], tf.int32)
#
#     y_pred = tf.reshape(y_pred, [-1, pred_shape[0], pred_shape[1], N_ANCHORS, N_CLASSES + 5])
#
#     # Create off set map
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
#     c_xy = tf.to_float(c_xy)
#
#     # Scale absolute predictions to relative values by dividing by output_size
#     output_size    = tf.cast(tf.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]), tf.float32)
#     anchors_tensor = np.reshape(self.anchors, [1, 1, 1, N_ANCHORS, 2])
#
#     pred_box_xy   = (tf.sigmoid(y_pred[:, :, :, :, :2]) + c_xy) / output_size
#     pred_box_wh   = tf.exp(y_pred[:, :, :, :, 2:4]) * anchors_tensor / output_size
#     pred_box_wh   = tf.sqrt(pred_box_wh)
#     pred_box_conf = tf.sigmoid(y_pred[:, :, :, :, 4:5])
#     pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])
#
#     # adjust confidence
#     pred_tem_wh   = tf.pow(pred_box_wh, 2) * output_size
#     pred_box_ul   = pred_box_xy - 0.5 * pred_tem_wh
#     pred_box_bd   = pred_box_xy + 0.5 * pred_tem_wh
#     pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
#
#     # Adjust ground truth
#     gt_shape = K.shape(y_true)  # shape of ground truth value
#     y_true = tf.reshape(y_true, [-1, gt_shape[1], gt_shape[2], N_ANCHORS, N_CLASSES + 5])
#
#     true_box_xy = y_true[:, :, :, :, 0:2]
#     true_box_wh = tf.sqrt(y_true[:, :, :, :, 2:4])
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
#     # This is confusing!! :(
#
#     # intersection over union
#     iou = tf.truediv(intersect_area,
#                      true_box_area + pred_box_area - intersect_area)
#
#     # For each cell, find the anchor has the highest IoU and set to True
#     best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
#     best_box = tf.to_float(best_box)
#     # Filter out other anchors in a given cell to zero. We only consider,
#     # highest IoU to compute the boxes
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
#     # Object Confidence Loss
#     weight_conf   = 0.5 * (1. - true_box_conf) + 5.0 * true_box_conf
#     obj_conf_loss = tf.pow(true_box_conf - pred_box_conf, 2) * weight_conf
#     obj_conf_loss = tf.reshape(obj_conf_loss, [-1, tf.cast(GRID_W * GRID_H, tf.int32) * N_ANCHORS])
#     obj_conf_loss = tf.reduce_mean(tf.reduce_sum(obj_conf_loss, 1))
#
#     # Classification Loss
#     weight_prob         = 1.0 * tf.concat(N_CLASSES * [true_box_conf], 4)
#     classification_loss = tf.pow(true_box_prob - pred_box_prob, 2) * weight_prob
#     classification_loss = tf.reshape(classification_loss,
#                                      [-1, tf.cast(GRID_W * GRID_H, tf.int32) * N_ANCHORS * N_CLASSES])
#     classification_loss = tf.reduce_mean(tf.reduce_sum(classification_loss, 1))
#
#     loss = 0.5 * (loc_loss + obj_conf_loss + classification_loss)
#     return loss
