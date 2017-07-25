import keras.backend as K
import tensorflow as tf
from cfg import *


def custom_loss(y_true, y_pred):
    """
    Loss Function of YOLOv2
    :param y_true: a Tensor  [batch_size, GRID_H, GRID_W, N_ANCHORS*(N_CLASSES + 5)] 
    :param y_pred: a Tensor [bacth_size, GRID_H, GRID_H, N_ANCHOR*(N_CLASSES + 5)]

    :return: a scalar
            loss value
    """
    pred_shape = K.shape(y_pred)[1:3]
    gt_shape = K.shape(y_true)  # shape of ground truth value
    GRID_H = tf.cast(pred_shape[0], tf.int32)  # shape of output feature map
    GRID_W = tf.cast(pred_shape[1], tf.int32)

    output_size = tf.cast(tf.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]), tf.float32)
    y_pred = tf.reshape(y_pred, [-1, pred_shape[0], pred_shape[1], N_ANCHORS, N_CLASSES + 5])
    y_true = tf.reshape(y_true, [-1, gt_shape[1], gt_shape[2], N_ANCHORS, N_CLASSES + 5])

    # Grid Map to calculate offset
    c_xy = _create_offset_map(K.shape(y_pred))

    # Scale anchors to correct aspect ratio
    # Extract prediction output from network
    pred_box_xy = (tf.sigmoid(y_pred[:, :, :, :, :2]) + c_xy) / output_size
    pred_box_wh = tf.exp(y_pred[:, :, :, :, 2:4]) * np.reshape(ANCHORS, [1, 1, 1, N_ANCHORS, 2])
    pred_box_wh = tf.sqrt(pred_box_wh / output_size)
    # adjust confidence
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)
    # adjust probability @TODO: create hierarchical soft-max tree
    pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])

    y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)

    # Adjust ground truth
    center_xy = y_true[:, :, :, :, 0:2]
    true_box_xy = center_xy
    true_box_wh = tf.sqrt(y_true[:, :, :, :, 2:4])

    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * output_size
    pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh

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
    true_box_prob = y_true[:, :, :, :, 5:]

    y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)

    # Compute the weights
    weight_coor = tf.concat(4 * [true_box_conf], 4)
    weight_coor = 5.0 * weight_coor
    weight_conf = 0.5 * (1. - true_box_conf) + 5.0 * true_box_conf
    weight_prob = tf.concat(N_CLASSES * [true_box_conf], 4)
    weight_prob = 1.0 * weight_prob

    weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)

    # Finalize the loss
    loss = tf.pow(y_pred - y_true, 2)
    loss = loss * weight
    loss = tf.reshape(loss, [-1, tf.cast(GRID_W * GRID_H, tf.int32) * N_ANCHORS * (4 + 1 + N_CLASSES)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)

    return loss


def _create_offset_map(output_shape):
    """
    In Yolo9000 paper, grid map
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


def avg_iou(y_true, y_pred):
    pred_shape = K.shape(y_pred)[1:3]
    gt_shape = K.shape(y_true)  # shape of ground truth value
    GRID_H = tf.cast(pred_shape[0], tf.int32)  # shape of output feature map
    GRID_W = tf.cast(pred_shape[1], tf.int32)

    output_size = tf.cast(tf.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]), tf.float32)
    y_pred = tf.reshape(y_pred, [-1, pred_shape[0], pred_shape[1], N_ANCHORS, N_CLASSES + 5])
    y_true = tf.reshape(y_true, [-1, gt_shape[1], gt_shape[2], N_ANCHORS, N_CLASSES + 5])
    c_xy = _create_offset_map(K.shape(y_pred))

    pred_box_xy = (tf.sigmoid(y_pred[..., :2]) + c_xy) / output_size
    pred_box_wh = tf.exp(y_pred[:, :, :, :, 2:4]) * np.reshape(ANCHORS, [1, 1, 1, N_ANCHORS, 2])
    pred_box_wh = tf.sqrt(pred_box_wh / output_size)

    # Adjust ground truth
    center_xy = y_true[:, :, :, :, 0:2]
    true_box_xy = center_xy
    true_box_wh = tf.sqrt(y_true[:, :, :, :, 2:4])

    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * output_size
    pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh

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
    return tf.reduce_sum(iou) / tf.to_float(gt_shape[0])


def recall(y_true, y_pred):
    pred_shape = K.shape(y_pred)[1:3]
    gt_shape = K.shape(y_true)  # shape of ground truth value
    GRID_H = tf.cast(pred_shape[0], tf.int32)  # shape of output feature map
    GRID_W = tf.cast(pred_shape[1], tf.int32)

    output_size = tf.cast(tf.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]), tf.float32)
    y_pred = tf.reshape(y_pred, [-1, pred_shape[0], pred_shape[1], N_ANCHORS, N_CLASSES + 5])
    y_true = tf.reshape(y_true, [-1, gt_shape[1], gt_shape[2], N_ANCHORS, N_CLASSES + 5])

    c_xy = _create_offset_map(K.shape(y_pred))

    pred_box_xy = (tf.sigmoid(y_pred[..., :2]) + c_xy) / output_size
    pred_box_wh = tf.exp(y_pred[:, :, :, :, 2:4]) * np.reshape(ANCHORS, [1, 1, 1, N_ANCHORS, 2])
    pred_box_wh = tf.sqrt(pred_box_wh / output_size)

    # Adjust ground truth
    center_xy = y_true[:, :, :, :, 0:2]
    true_box_xy = center_xy
    true_box_wh = tf.sqrt(y_true[:, :, :, :, 2:4])

    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * output_size
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh

    true_tem_wh = tf.pow(true_box_wh, 2) * output_size
    true_box_area = true_tem_wh[:, :, :, :, 0] * true_tem_wh[:, :, :, :, 1]
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh

    intersect_ul = tf.maximum(pred_box_ul, true_box_ul)
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    recall_value = tf.truediv(intersect_area, true_box_area)
    return tf.reduce_sum(recall_value)


def precision(y_true, y_pred):
    pred_shape = K.shape(y_pred)[1:3]
    gt_shape = K.shape(y_true)  # shape of ground truth value
    GRID_H = tf.cast(pred_shape[0], tf.int32)  # shape of output feature map
    GRID_W = tf.cast(pred_shape[1], tf.int32)

    output_size = tf.cast(tf.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]), tf.float32)
    y_pred = tf.reshape(y_pred, [-1, pred_shape[0], pred_shape[1], N_ANCHORS, N_CLASSES + 5])
    y_true = tf.reshape(y_true, [-1, gt_shape[1], gt_shape[2], N_ANCHORS, N_CLASSES + 5])

    c_xy = _create_offset_map(K.shape(y_pred))

    pred_box_xy = (tf.sigmoid(y_pred[..., :2]) + c_xy) / output_size
    pred_box_wh = tf.exp(y_pred[:, :, :, :, 2:4]) * np.reshape(ANCHORS, [1, 1, 1, N_ANCHORS, 2])
    pred_box_wh = tf.sqrt(pred_box_wh / output_size)

    # Adjust ground truth
    center_xy = y_true[:, :, :, :, 0:2]
    true_box_xy = center_xy
    true_box_wh = tf.sqrt(y_true[:, :, :, :, 2:4])

    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * output_size
    pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh

    true_tem_wh = tf.pow(true_box_wh, 2) * output_size
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh

    intersect_ul = tf.maximum(pred_box_ul, true_box_ul)
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

    precision_value = tf.truediv(intersect_area, pred_box_area)
    return tf.reduce_sum(precision_value) / tf.to_float(gt_shape[0])