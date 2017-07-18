import keras.backend as K
import tensorflow as tf
from cfg import *


def custom_loss(y_true, y_pred):
    """
    y_true: shape [BATCH, OUTPUT_W, OUTPUT_H, N_ANCHORS*(N_CLASSES+5]
    y_pred: [BATCH, OUTPUT_W, OUTPUT_H, N_ANCHORS, N_CLASSES + 5]

    y_true provides: [xc, yc, w, h, conf, class_prob]
    y
    """
    pred_shape = K.shape(y_pred)[1:3]

    true_boxes, true_conf, true_cls = _process_gt(y_true)
    pred_boxes, pred_conf, pred_cls = _process_prediction(y_pred)

    # Calculate IoU
    iou = _calc_iou(true_boxes, pred_boxes)

    # Create detection mask
    best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
    best_box = tf.to_float(best_box)
    true_conf = tf.expand_dims(best_box * true_conf, -1)  # sigmoid(to) = P(object) * IoU
    pred_conf = tf.expand_dims(pred_conf, -1)

    # Weight Terms
    weight_coor = 5.0 * tf.concat(4 * [true_conf], 4)
    weight_conf = 5.0 * true_conf + 0.1* (1. - true_conf)
    weight_prob = 5.0 * tf.concat(N_CLASSES * [true_conf], 4)
    weight_terms = tf.concat([weight_coor, weight_conf, weight_prob], 4)

    # Update wh --- > sqrt(wh)
    gt_boxes = tf.concat([true_boxes[..., :2], tf.sqrt(true_boxes[..., 2:4])], 4)
    pred_boxes = tf.concat([pred_boxes[..., :2], tf.sqrt(pred_boxes[..., 2:4])], 4)

    y_true = tf.concat([gt_boxes, true_conf, true_cls], 4)
    y_pred = tf.concat([pred_boxes, pred_conf, pred_cls], 4)

    # Total loss
    loss = tf.pow(y_pred - y_true, 2)
    loss = loss * weight_terms
    loss = tf.reshape(loss, [-1, N_ANCHORS * (5 + N_CLASSES) * K.cast(pred_shape[0] * pred_shape[1], tf.int32)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)

    return loss


def _process_gt(y_true):
    """
    Process ground truth output
    """
    gt_shape = K.shape(y_true)
    y_true = K.reshape(y_true, [gt_shape[0], gt_shape[1], gt_shape[2], N_ANCHORS, N_CLASSES + 5])
    true_boxes = y_true[..., :4]
    true_conf = y_true[..., 4]
    true_clf = y_true[..., 5:]
    return true_boxes, true_conf, true_clf


def _process_prediction(y_pred):
    output_shape = K.shape(y_pred)[1:3]
    OUTPUT_H = tf.cast(output_shape[0], tf.int32)
    OUTPUT_W = tf.cast(output_shape[1], tf.int32)

    # Scaled anchors to size of feature map
    scaled_anchors = ANCHORS * K.cast(K.reshape([OUTPUT_W, OUTPUT_H], [1, 1, 1, 1, 2]), tf.float32)
    anchor_tensor = K.reshape(scaled_anchors, [1, 1, 1, N_ANCHORS, 2])
    y_pred = K.reshape(y_pred, [-1, output_shape[0], output_shape[1], N_ANCHORS, N_CLASSES + 5])

    # Create offset map
    cx, cy = _create_offset_map(K.shape(y_pred))
    px = tf.cast(anchor_tensor[..., 0], dtype=tf.float32)
    py = tf.cast(anchor_tensor[..., 1], dtype=tf.float32)

    # Calculate Prediction in relative position (percentage)    
    bx = (tf.sigmoid(y_pred[..., 0]) + cx) / tf.cast(OUTPUT_W, tf.float32)
    by = (tf.sigmoid(y_pred[..., 1]) + cy) / tf.cast(OUTPUT_H, tf.float32)
    bw = px * tf.exp(y_pred[..., 2]) / tf.cast(OUTPUT_W, tf.float32)
    bh = py * tf.exp(y_pred[..., 3]) / tf.cast(OUTPUT_H, tf.float32)

    # Extract features from prediction
    pred_boxes = tf.stack([bx, by, bw, bh], -1)
    pred_conf = tf.sigmoid(y_pred[..., 4])  # to = sig
    pred_clf = tf.nn.softmax(y_pred[..., 5:])

    return pred_boxes, pred_conf, pred_clf


def _calc_iou(true_boxes, pred_boxes):
    # Scaled anchors to size of feature map
    output_shape = K.shape(pred_boxes)[1:3]

    # Scale to input image
    GRID_SIZE = K.cast(K.reshape([output_shape[1], output_shape[0]], [1, 1, 1, 1, 2]), tf.float32)

    pred_xy = pred_boxes[..., :2] * GRID_SIZE
    pred_wh = pred_boxes[..., 2:4] * GRID_SIZE
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]

    true_xy = true_boxes[..., :2] * GRID_SIZE
    true_wh = true_boxes[..., 2:4] * GRID_SIZE
    true_area = true_wh[..., 0] * true_wh[..., 1]

    # Calculate IoU between ground truth and prediction
    intersect_ul = tf.maximum(pred_xy - 0.5 * pred_wh, true_xy - 0.5 * true_wh)
    intersect_br = tf.minimum(pred_xy + 0.5 * pred_wh, true_xy + 0.5 * true_wh)
    intersect_wh = tf.maximum(intersect_br - intersect_ul, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    iou = tf.truediv(intersect_area, true_area + pred_area - intersect_area)

    return iou


def _create_offset_map(output_shape):
    """
    In Yolo9000 paper, grid map
    """
    GRID_H = tf.cast(output_shape[1], tf.int32)
    GRID_W = tf.cast(output_shape[2], tf.int32)
    N_ANCHORS = len(ANCHORS)

    cx = tf.cast((K.arange(0, stop=GRID_W)), dtype=tf.float32)
    cx = K.expand_dims(cx, -1)
    cx = K.tile(cx, (GRID_H, N_ANCHORS))
    cx = K.reshape(cx, [-1, GRID_H, GRID_W, N_ANCHORS])

    cy = K.cast((K.arange(0, stop=GRID_H)), dtype=tf.float32)
    cy = K.reshape(cy, [-1, 1])
    cy = K.tile(cy, [1, N_ANCHORS * GRID_W])
    cy = K.reshape(cy, [-1])
    cy = K.reshape(cy, [-1, GRID_H, GRID_W, N_ANCHORS])

    return cx, cy

