"""
Loss (Objective) Function definition for You Only Look Once version 2
"""
import numpy as np
import tensorflow as tf
import keras.backend as K


def yolov2_loss(anchors, num_classes):

    def compute_loss(y_true, y_pred):
        output_dims = tf.shape(y_pred)
        height = output_dims[1]
        width = output_dims[2]

        pred_boxes = y_pred[..., 0:4]
        pred_conf = y_pred[..., 4]
        pred_classes = y_pred[..., 5:]

        gt_boxes = y_true[..., 0:4]
        gt_conf = y_true[..., 4]
        gt_classes = y_true[..., 5:]

        obj_cells = tf.expand_dims(y_true[..., 4], -1)
        noobj_cells = tf.expand_dims(1.0 - y_true[..., 4], -1)

        #  Create offset map
        cx = tf.reshape(tf.tile(tf.range(width), [height]), [-1, height, width, 1])
        cy = tf.tile(tf.expand_dims(tf.range(height), -1), [1, width])
        cy = tf.reshape(cy, [-1, height, width, 1])
        c_xy = tf.to_float(tf.stack([cx, cy], -1))

        with tf.name_scope('RegressionLoss'):
            # Convert [y1, x1, y2, x2] to [y_c, x_c, h, w] coordinates
            gt_wh = gt_boxes[..., 2:4] - gt_boxes[..., 0:2]
            pred_wh = pred_boxes[..., 2:4] - pred_boxes[..., 0:2]
            gt_centroids = gt_boxes[..., 0:2] + gt_wh
            pred_centroids = pred_boxes[..., 0:2] + pred_wh

            grid_centroids = c_xy + 0.5
            grid_wh = tf.cast(tf.reshape(anchors, [1, 1, 1, len(anchors), 2]), tf.float32)

            obj_coord   = obj_cells   * (tf.square(pred_centroids - gt_centroids) +
                                         tf.square(tf.sqrt(pred_wh) - tf.sqrt(gt_wh)))

            noobj_coord = noobj_cells * (tf.square(pred_centroids - grid_centroids) +
                                         tf.square(tf.sqrt(pred_wh) - tf.sqrt(tf.ones_like(pred_wh) * grid_wh)))

            coord_loss = 1.0 * obj_coord + 0.01 * noobj_coord

        with tf.name_scope('ObjectConfidenceLoss'):
            box_iou = compute_iou(gt_boxes, pred_boxes)
            best_iou = tf.expand_dims(tf.reduce_max(box_iou, axis=-1), -1)

            obj_conf   = obj_cells   * tf.expand_dims(tf.square(pred_conf - gt_conf * box_iou), -1)
            noobj_conf = noobj_cells * tf.expand_dims(tf.square(pred_conf - 0.0) * tf.to_float(best_iou < 0.6), -1)

            conf_loss = 5.0 * obj_conf + 0.01 * noobj_conf

        # # @TODO: focal loss
        with tf.name_scope("ClassificationLoss"):
            cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=gt_classes, logits=pred_classes)
            cls_loss = obj_cells * tf.expand_dims(cls_loss, -1)

        total_loss = tf.reduce_sum(coord_loss) + tf.reduce_sum(conf_loss) + tf.reduce_sum(cls_loss)
        return total_loss

    return compute_loss


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

# mask_shape = tf.shape(y_true)[:4]
#
# cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
# cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
#
# cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, 5, 1])
#
# coord_mask = tf.zeros(mask_shape)
# conf_mask = tf.zeros(mask_shape)
# class_mask = tf.zeros(mask_shape)
#
# seen = tf.Variable(0.)
# total_recall = tf.Variable(0.)
#
# """
# Adjust prediction
# """
# ### adjust x and y
# pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
#
# ### adjust w and h
# pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2])
#
# ### adjust confidence
# pred_box_conf = tf.sigmoid(y_pred[..., 4])
#
# ### adjust class probabilities
# pred_box_class = y_pred[..., 5:]
#
# """
# Adjust ground truth
# """
# ### adjust x and y
# true_box_xy = y_true[..., 0:2]  # relative position to the containing cell
#
# ### adjust w and h
# true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically
#
# ### adjust confidence
# true_wh_half = true_box_wh / 2.
# true_mins = true_box_xy - true_wh_half
# true_maxes = true_box_xy + true_wh_half
#
# pred_wh_half = pred_box_wh / 2.
# pred_mins = pred_box_xy - pred_wh_half
# pred_maxes = pred_box_xy + pred_wh_half
#
# intersect_mins = tf.maximum(pred_mins, true_mins)
# intersect_maxes = tf.minimum(pred_maxes, true_maxes)
# intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
# intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
#
# true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
# pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]
#
# union_areas = pred_areas + true_areas - intersect_areas
# iou_scores = tf.truediv(intersect_areas, union_areas)
#
# true_box_conf = iou_scores * y_true[..., 4]
#
# ### adjust class probabilities
# true_box_class = tf.argmax(y_true[..., 5:], -1)
#
# """
# Determine the masks
# """
# ### coordinate mask: position of the ground truth boxes (the predictors)
# coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale
#
# ### confidence mask: penalize predictors + penalize boxes with low IOU
# # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6

# iou_scores = tf.truediv(intersect_areas, union_areas)
# best_ious = tf.reduce_max(iou_scores, axis=4)

# conf_mask =  no_obj_scale * (1 - y_true[..., 4]) * tf.to_float(best_ious < 0.6)+
#              obj_scale    * y_true[..., 4]
#
# ### class mask:
# class_mask = class_scale * y_true[..., 4] * tf.gather(self.class_wt, true_box_class)
#
# """
# Warm-up training
# """
# no_boxes_mask = tf.to_float(coord_mask < self.coord_scale / 2.)
# seen = tf.assign_add(seen, 1.)
#
# true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_bs),
#                                                lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
#                                                         true_box_wh + tf.ones_like(true_box_wh) * np.reshape(
#                                                             self.anchors, [1, 1, 1, self.nb_box, 2]) * no_boxes_mask,
#                                                         tf.ones_like(coord_mask)],
#                                                lambda: [true_box_xy,
#                                                         true_box_wh,
#                                                         coord_mask])
#
# """
# Finalize the loss
# """
# nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
# nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
# nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
#
# loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
# loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
# loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
# loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
# loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
#
# loss = loss_xy + loss_wh + loss_conf + loss_class
##