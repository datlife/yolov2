from cfg import *
import keras.backend as K
import tensorflow as tf

def custom_loss(y_true, y_pred):
    
    pred_shape = K.shape(y_pred)[1:3]
    
    GRID_W = tf.cast(pred_shape[0], tf.float32)
    GRID_H = tf.cast(pred_shape[1], tf.float32)
    
    NORM_W = tf.cast(32.*GRID_W, tf.float32)
    NORM_H = tf.cast(32.*GRID_H, tf.float32)
    
    no_object_scale = 0.5
    object_scale     = 5.0
    coordinate_scale = 1.0
    class_scale      = 1.0
    N_ANCHORS        = len(ANCHORS)

    pred_shape = K.shape(y_pred)[1:3]
    y_pred = K.reshape(y_pred, [-1, pred_shape[0], pred_shape[1], N_ANCHORS, N_CLASSES + 5])
    y_true = K.reshape(y_true, [-1, pred_shape[0], pred_shape[1], N_ANCHORS, N_CLASSES + 5])

    anchor_tensor = np.reshape(ANCHORS, [1, 1, 1, N_ANCHORS, 2])

    #  Adjust Prediction
    pred_box_xy   = tf.sigmoid(y_pred[:, :, :, :, :2])
    pred_box_wh   = tf.exp(y_pred[:, :, :, :, 2:4]) * anchor_tensor
    pred_box_wh   = tf.sqrt(pred_box_wh / K.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]))
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)  # adjust confidence
    pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])  # adjust probability

    #  Adjust ground truth
    center_xy = y_true[:, :, :, :, 0:2]
    center_xy = center_xy / K.reshape([(NORM_W / GRID_W), (NORM_H / GRID_H)], [1, 1, 1, 1, 2])

    true_box_xy = center_xy - tf.floor(center_xy)
    true_box_wh = y_true[:, :, :, :, 2:4]
    true_box_wh = tf.sqrt(true_box_wh / K.reshape([NORM_W, NORM_H], [1, 1, 1, 1, 2]))

    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * K.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2])
    pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh

    true_tem_wh = tf.pow(true_box_wh, 2) * K.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2])
    true_box_area = true_tem_wh[:, :, :, :, 0] * true_tem_wh[:, :, :, :, 1]
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh

    # Calculate IoU between ground truth and prediction
    intersect_ul = tf.maximum(pred_box_ul, true_box_ul)
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

    iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
    best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
    best_box = tf.to_float(best_box)
    # calculate avg_iou here

    true_box_conf = tf.expand_dims(best_box * y_true[:, :, :, :, 4], -1)
    true_box_prob = y_true[:, :, :, :, 5:]  # adjust confidence

    y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)
    y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)

    # Compute the weights

    # Object Confidence Loss
    weight_conf = no_object_scale * (1. - true_box_conf) + object_scale * true_box_conf

    # Object Localization Loss
    weight_coor = tf.concat(4 * [true_box_conf], 4)
    weight_coor = coordinate_scale * weight_coor

    # Object Classification Loss
    weight_prob = tf.concat(N_CLASSES * [true_box_conf], 4)
    weight_prob = class_scale * weight_prob

    # Total Loss
    weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)

    # ## Finalize the loss
    loss = tf.pow(y_pred - y_true, 2)
    loss = loss * weight
    loss = tf.reshape(loss, [-1, tf.cast(GRID_W * GRID_H * N_ANCHORS * (4 + 1 + N_CLASSES), tf.int32)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)

    return loss

