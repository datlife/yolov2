"""
Loss (Objective) Function definition for You Only Look Once version 2
"""
import numpy as np
import tensorflow as tf
import keras.backend as K


def yolov2_loss(anchors, num_classes):

    def compute_loss(y_true, y_pred):

        pred_boxes   = y_pred[..., 0:4]
        pred_conf    = y_pred[..., 4]
        pred_classes = y_pred[..., 5:]

        gt_boxes   = y_true[..., 0:4]
        gt_conf    = y_true[..., 4]
        gt_classes = y_true[..., 5:]

        # IOU scores shape [..., 5, 1]
        box_iou = compute_iou(gt_boxes, pred_boxes)
        box_iou = tf.reduce_max(box_iou, axis=-1)

        true_conf = gt_conf * box_iou

        return true_conf
        # # @TODO: focal loss
        # loc_loss  = compute_localization_loss(gt_boxes, pred_boxes)
        # conf_loss = tf.reduce_sum(tf.square(pred_conf - gt_conf * box_iou))
        # cls_loss  = compute_classification_loss(gt_classes, pred_classes)
        #
        # total_loss = loc_loss + conf_loss + cls_loss
        # return total_loss
        # return total_loss

    def compute_localization_loss(gt_boxes, pred_boxes):
        xy_loss = tf.square(pred_boxes[..., 0:2] - gt_boxes[..., 0:2])
        xy_loss = tf.reduce_sum(xy_loss)

        wh_loss = tf.square(tf.sqrt(pred_boxes[..., 0:2]) - tf.sqrt(gt_boxes[..., 0:2]))
        wh_loss = tf.reduce_sum(wh_loss)

        return xy_loss + wh_loss

    def compute_classification_loss(gt_classes, pred_classes):
        cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=gt_classes, logits=pred_classes)
        cls_loss = tf.reduce_sum(cls_loss)
        return cls_loss

    return compute_loss


def compute_iou(gt_boxes, pred_boxes, scope=None):
    with tf.name_scope(scope, 'IoU'):
        with tf.name_scope('area'):
            gt_wh   = gt_boxes[..., 2:4]   - gt_boxes[..., 0:2]
            pr_wh   = pred_boxes[..., 2:4] - pred_boxes[..., 0:2]

            gt_area = gt_wh[..., 0] * gt_wh[..., 1]
            pr_area = pr_wh[..., 0] * pr_wh[..., 1]

        with tf.name_scope('intersections'):
            intersect_mins  = tf.maximum(gt_boxes[..., 0:2], pred_boxes[..., 0:2])
            intersect_maxes = tf.minimum(gt_boxes[..., 2:4], pred_boxes[..., 2:4])
            intersect_hw    = tf.maximum(intersect_maxes - intersect_mins, 0.)
            intersections   = intersect_hw[..., 0] * intersect_hw[..., 1]

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
# return loss
# #
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
