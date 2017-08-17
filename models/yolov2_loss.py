"""
Loss of YOLOv2 Implementation. Few Notes
    * What we get from the CNN is a feature map (imagine as a 3-D box)
    * Each cell in a feature map is a vector size N_ANCHORS* (5 + N_CLASSES)  as:
            eg. one cell
            ------------- ANCHOR 1---------------  -------- ANCHORS 2 -------------  ...... ------------ANCHOR N -----------
            [tx, ty, tw, th, to , label_vector..],[tx1, ty1, tw1, th1, label_vector]......[tx_n, ty_n, tw_n, th_m, label...]
            * tx, ty : predicts of relative center of bounding box to its current cell. Therefore, true center points of
              a prediction would be :
                        xc = sigmoid(tx) + cx
                        yc = sigmoid(ty) + cy
            * tw, th: predicts the scaling value for true width and height of the bounding box based on the anchor as:
                        w   = exp(tw) * px
                        h   = exp(th) * py
            * to : objectiveness of the cell : the probability of having an object in the cell
            * label_vector: classification vector to calculate soft-max
"""
import numpy as np
import tensorflow as tf
import keras.backend as K
from cfg import *


def custom_loss(y_true, y_pred):
    """
    Loss Function of YOLOv2
    :param y_true: a Tensor  [batch_size, GRID_H, GRID_W, N_ANCHORS*(N_CLASSES + 5)]
    :param y_pred: a Tensor  [batch_size, GRID_H, GRID_H, N_ANCHOR*(N_CLASSES + 5)]
    :return: a scalar
            loss value
    """
    pred_shape = K.shape(y_pred)[1:3]
    gt_shape = K.shape(y_true)                 # shape of ground truth value
    GRID_H = tf.cast(pred_shape[0], tf.int32)  # shape of output feature map
    GRID_W = tf.cast(pred_shape[1], tf.int32)

    output_size = tf.cast(tf.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]), tf.float32)
    y_pred = tf.reshape(y_pred, [-1, pred_shape[0], pred_shape[1], N_ANCHORS, N_CLASSES + 5])
    y_true = tf.reshape(y_true, [-1, gt_shape[1], gt_shape[2], N_ANCHORS, N_CLASSES + 5])

    # Grid Map to calculate offset
    c_xy = _create_offset_map(K.shape(y_pred))

    # Extract prediction output from network
    pred_xy   = (tf.sigmoid(y_pred[:, :, :, :, :2]) + c_xy) / output_size
    pred_wh   = (tf.exp(y_pred[:, :, :, :, 2:4]) * np.reshape(ANCHORS, [1, 1, 1, N_ANCHORS, 2])) / output_size
    pred_wh   = tf.sqrt(pred_wh)
    pred_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)
    pred_cls  = y_pred[:, :, :, :, 5:]

    # Adjust ground truth
    true_box_xy = y_true[:, :, :, :, 0:2]
    true_box_wh = tf.sqrt(y_true[:, :, :, :, 2:4])

    # adjust confidence
    pred_tem_wh = tf.pow(pred_wh, 2) * output_size
    pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
    pred_box_ul = pred_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_xy + 0.5 * pred_tem_wh

    # Calculate IoU between prediction and ground truth
    true_tem_wh = tf.pow(true_box_wh, 2) * output_size
    true_box_area = true_tem_wh[:, :, :, :, 0] * true_tem_wh[:, :, :, :, 1]
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh

    intersect_ul = tf.maximum(pred_box_ul, true_box_ul)
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

    iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
    best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
    best_box = tf.to_float(best_box)
    true_box_conf = tf.expand_dims(best_box * y_true[:, :, :, :, 4], -1)

    # adjust confidence
    pred_boxes = tf.concat([pred_xy, pred_wh], 4)
    true_boxes = tf.concat([true_box_xy, true_box_wh], 4)

    # Coordinate Loss
    weight_coor     = 5.0 * tf.concat(4 * [true_box_conf], 4)
    coordinate_loss = tf.pow(pred_boxes - true_boxes, 2) * weight_coor
    coordinate_loss = tf.reshape(coordinate_loss, [-1, tf.cast(GRID_W * GRID_H, tf.int32) * N_ANCHORS * 4])
    coordinate_loss = tf.reduce_sum(coordinate_loss, 1)
    coordinate_loss = tf.reduce_mean(coordinate_loss)

    # Object Confidence loss
    weight_conf = 0.5 * (1. - true_box_conf) + 5.0 * true_box_conf
    conf_loss = tf.pow(true_box_conf - pred_conf, 2) * weight_conf
    conf_loss = tf.reshape(conf_loss, [-1, tf.cast(GRID_W * GRID_H, tf.int32) * N_ANCHORS * 1])
    conf_loss = tf.reduce_sum(conf_loss, 1)
    conf_loss = tf.reduce_mean(conf_loss)

    # Category probability Loss
    weight_prob    = 1.0 * tf.concat(N_CLASSES * [true_box_conf], 4)
    category_prob  = (pred_conf * pred_cls) * weight_prob
    category_loss  = tf.nn.softmax_cross_entropy_with_logits(labels=y_true[..., 5:], logits=category_prob)
    category_loss  = tf.reduce_mean(category_loss)

    # Total loss
    total_loss = 0.5 * (coordinate_loss + conf_loss + category_loss)
    return total_loss


def _create_offset_map(output_shape):
    """
    In Yolo9000 paper, Grid map to calculate offsets for each cell in the output feature map
    """
    GRID_H = tf.cast(output_shape[1], tf.int32)  # shape of output feature map
    GRID_W = tf.cast(output_shape[2], tf.int32)

    cx = tf.cast((K.arange(0, stop=GRID_W)), dtype=tf.float32)
    cx = K.tile(cx, [GRID_H])
    cx = K.reshape(cx, [-1, GRID_H, GRID_W, 1])

    cy = K.cast((K.arange(0, stop=GRID_H)), dtype=tf.float32)
    cy = K.reshape(cy, [-1, 1])
    cy = K.tile(cy, [1, GRID_W])
    cy = K.reshape(cy, [-1])
    cy = K.reshape(cy, [-1, GRID_H, GRID_W, 1])

    c_xy = tf.stack([cx, cy], -1)
    c_xy = K.cast(c_xy, tf.float32)

    return c_xy