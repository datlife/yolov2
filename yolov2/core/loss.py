"""
Loss (Objective) Function definition for You Only Look Once version 2
"""
import numpy as np
import tensorflow as tf
import keras.backend as K


class YOLOV2Loss(object):

    def __init__(self, anchors, num_classes, warm_up_steps=100):
        self.anchors = anchors
        self.num_classes = num_classes
        self.warm_up_steps = warm_up_steps
        self.curr_epochs = 0

    def compute_loss(self, y_true, y_pred):

        output_dims = tf.shape(y_pred)
        height = output_dims[1]
        width = output_dims[2]

        pred_boxes   = y_pred[..., 0:4]
        pred_conf    = y_pred[..., 4]
        pred_classes = y_pred[..., 5:]

        gt_boxes   = y_true[..., 0:4]
        gt_conf    = y_true[..., 4]
        gt_classes = y_true[..., 5:]

        obj_cells   = tf.expand_dims(y_true[..., 4], -1)
        noobj_cells = tf.expand_dims(1.0 - y_true[..., 4], -1)

        #  Create offset map
        cx   = tf.reshape(tf.tile(tf.range(width), [height]), [-1, height, width, 1])
        cy   = tf.tile(tf.expand_dims(tf.range(height), -1), [1, width])
        cy   = tf.reshape(cy, [-1, height, width, 1])
        c_xy = tf.to_float(tf.stack([cx, cy], -1))

        with tf.name_scope('RegressionLoss'):

            # Convert [y1, x1, y2, x2] to [y_c, x_c, h, w] coordinates
            gt_wh          = gt_boxes[..., 2:4]   - gt_boxes[..., 0:2]
            pred_wh        = pred_boxes[..., 2:4] - pred_boxes[..., 0:2]
            gt_centroids   = gt_boxes[..., 0:2]   + (gt_wh / 2.0)
            pred_centroids = pred_boxes[..., 0:2] + (pred_wh / 2.0)

            grid_centroids = c_xy + 0.5
            grid_wh = tf.cast(tf.reshape(self.anchors, [1, 1, 1, len(self.anchors), 2]), tf.float32)

            obj_coord    = obj_cells * (tf.square(pred_centroids - gt_centroids) +
                                        tf.square(tf.sqrt(pred_wh) - tf.sqrt(gt_wh)))

            no_obj_coord = noobj_cells * (tf.square(pred_centroids - grid_centroids) +
                                          tf.square(tf.sqrt(pred_wh) - tf.sqrt(tf.ones_like(pred_wh) * grid_wh)))

            coord_loss = 5.0 * obj_coord + 0.01 * no_obj_coord

        with tf.name_scope('ObjectConfidenceLoss'):
            box_iou = compute_iou(gt_boxes, pred_boxes)
            best_iou = tf.expand_dims(tf.reduce_max(box_iou, axis=-1), -1)

            obj_conf    = obj_cells   * tf.expand_dims(tf.square(pred_conf - gt_conf * box_iou), -1)
            no_obj_conf = noobj_cells * tf.expand_dims(tf.square(pred_conf - 0.0) * tf.to_float(best_iou < 0.6), -1)
            conf_loss = 1.0 * obj_conf + 0.01 * no_obj_conf

        with tf.name_scope("ClassificationLoss"):
            cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=gt_classes, logits=pred_classes)
            cls_loss = 1.0 * obj_cells * tf.expand_dims(cls_loss, -1)

        coord_loss = tf.Print(coord_loss, [tf.reduce_sum(coord_loss)], "[INFO] Coordinate Loss:")
        conf_loss = tf.Print(conf_loss, [tf.reduce_sum(conf_loss)], "[INFO] Confidence Loss:")
        cls_loss = tf.Print(cls_loss, [tf.reduce_sum(cls_loss)], "[INFO] Classfication Loss:")

        return tf.reduce_sum(coord_loss) + tf.reduce_sum(conf_loss) + tf.reduce_sum(cls_loss)
        # # @TODO: focal loss


def compute_iou(gt_boxes, pred_boxes, scope=None):
    with tf.name_scope(scope, 'IoU'):
        with tf.name_scope('area'):
            gt_wh = gt_boxes[..., 2:4] - gt_boxes[..., 0:2]
            pr_wh = pred_boxes[..., 2:4] - pred_boxes[..., 0:2]

            gt_area = gt_wh[..., 0] * gt_wh[..., 1]
            pr_area = pr_wh[..., 0] * pr_wh[..., 1]

        with tf.name_scope('intersections'):
            intersect_mins = tf.maximum(gt_boxes[..., 0:2], pred_boxes[..., 0:2])
            intersect_maxes = tf.minimum(gt_boxes[..., 2:4], pred_boxes[..., 2:4])
            intersect_hw = tf.maximum(intersect_maxes - intersect_mins, 0.)
            intersections = intersect_hw[..., 0] * intersect_hw[..., 1]

        unions = gt_area + pr_area - intersections
        iou = tf.where(tf.equal(intersections, 0.0),
                       tf.zeros_like(intersections),
                       tf.truediv(intersections, unions))

        return iou
