import tensorflow as tf
import keras.backend as K
from model.loss import _process_gt, _process_prediction, _calc_iou
def avg_iou(y_true, y_pred):
    pred_shape = K.shape(y_pred)[1:3]

    true_boxes, true_conf, true_cls = _process_gt(y_true)
    pred_boxes, pred_conf, pred_cls = _process_prediction(y_pred)
    
    iou           = _calc_iou(true_boxes, pred_boxes)
    true_conf     = tf.expand_dims(iou * true_conf, -1)   # sigmoid(to) = P(object) * IoU
    return tf.reduce_sum(true_conf)

def coor(y_true, y_pred):
    pred_shape = K.shape(y_pred)[1:3]

    true_boxes, true_conf, true_cls = _process_gt(y_true)
    pred_boxes, pred_conf, pred_cls = _process_prediction(y_pred)
    
    iou           = _calc_iou(true_boxes, pred_boxes)
    best_box      = tf.equal(iou, tf.reduce_max(iou, [3], True)) 
    best_box      = tf.to_float(best_box)
    true_conf     = tf.expand_dims(best_box * true_conf, -1)   # sigmoid(to) = P(object) * IoU
    pred_conf     = tf.expand_dims(pred_conf, -1)
    
    # Update wh --- > sqrt(wh)
    gt_boxes   = tf.concat([true_boxes[..., :2], tf.sqrt(true_boxes[...,2:4])], 4)
    pred_boxes = tf.concat([pred_boxes[..., :2], tf.sqrt(pred_boxes[...,2:4])], 4)  

    weight_coor = 5.0 * tf.concat(4 * [true_conf], 4)
    coor_loss = tf.pow(pred_boxes - gt_boxes, 2) * weight_coor
    return tf.reduce_sum(coor_loss)

def obj(y_true, y_pred):
    pred_shape = K.shape(y_pred)[1:3]

    true_boxes, true_conf, true_cls = _process_gt(y_true)
    pred_boxes, pred_conf, pred_cls = _process_prediction(y_pred)
    iou           = _calc_iou(true_boxes, pred_boxes)
    
    best_box      = tf.equal(iou, tf.reduce_max(iou, [3], True)) 
    best_box      = tf.to_float(best_box)
    true_conf     = tf.expand_dims(best_box * true_conf, -1)   # sigmoid(to) = P(object) * IoU
    pred_conf     = tf.expand_dims(pred_conf, -1)
    
    weight_conf = 5.0 * true_conf   + 0.1 * (1. - true_conf)
    obj_conf = tf.pow(pred_conf - true_conf, 2) * weight_conf
    
    return tf.reduce_sum(obj_conf)
