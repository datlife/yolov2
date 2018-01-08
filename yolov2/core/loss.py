"""
Loss (Objective) Function definition for You Only Look Once version 2
"""
import tensorflow as tf

EPSILON = 1e-8

class YOLOV2Loss(object):

  def __init__(self, anchors, num_classes, summary=True):
    self.anchors = anchors
    self.num_classes = num_classes
    self.summary = summary

  def compute_loss(self, y_true, y_pred):
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

    num_objects = tf.reduce_sum(tf.to_float(obj_cells > 0.0))
    num_no_objects = tf.reduce_sum(tf.to_float(noobj_cells > 0.0))

    with tf.name_scope('RegressionLoss'):
      # Convert [y1, x1, y2, x2] to [y_c, x_c, h, w] coordinates
      gt_wh = gt_boxes[..., 2:4] - gt_boxes[..., 0:2]
      pred_wh = pred_boxes[..., 2:4] - pred_boxes[..., 0:2]

      gt_centroids = gt_boxes[..., 0:2] + (gt_wh / 2.0)
      pred_centroids = pred_boxes[..., 0:2] + (pred_wh / 2.0)

      #  Create offset grid
      cx = tf.reshape(tf.tile(tf.range(width), [height]), [-1, height, width, 1])
      cy = tf.tile(tf.expand_dims(tf.range(height), -1), [1, width])
      cy = tf.reshape(cy, [-1, height, width, 1])

      grid_centroids = 0.5 + tf.to_float(tf.stack([cx, cy], -1))
      grid_anchors = tf.cast(tf.reshape(self.anchors, [1, 1, 1, len(self.anchors), 2]), tf.float32)

      # For cells containing objects, we penalize the difference between ground truths and predictions
      obj_coord = obj_cells * (tf.square(pred_centroids - gt_centroids) +
                               tf.square(tf.sqrt(pred_wh) - tf.sqrt(gt_wh)))

      # For cells containing no objects, we penalize the difference between predictions
      # and the centroids + anchors of that cells
      no_obj_coord = noobj_cells * (tf.square(pred_centroids - grid_centroids) +
                                    tf.square(tf.sqrt(pred_wh) - tf.sqrt(tf.ones_like(pred_wh) * grid_anchors)))
      # Average out
      obj_coord = tf.reduce_sum(obj_coord) / (num_objects + EPSILON)
      no_obj_coord = tf.reduce_sum(no_obj_coord) / (num_no_objects + EPSILON)

      coord_loss = 5.0 * obj_coord + 0.01 * no_obj_coord

    with tf.name_scope('ObjectConfidenceLoss'):
      box_iou = self.compute_iou(gt_boxes, pred_boxes)
      best_iou = tf.expand_dims(tf.reduce_max(box_iou, axis=-1), -1)

      obj_conf = obj_cells * tf.expand_dims(tf.square(pred_conf - gt_conf * box_iou), -1)
      no_obj_conf = noobj_cells * tf.expand_dims(tf.square(pred_conf - 0.0) * tf.to_float(best_iou < 0.6), -1)
      obj_conf = tf.reduce_sum(obj_conf) / (num_objects + EPSILON)
      no_obj_conf = tf.reduce_sum(no_obj_conf) / (num_no_objects + EPSILON)
      conf_loss = 1.0 * obj_conf + 0.01 * no_obj_conf

    # # @TODO: focal loss
    with tf.name_scope("ClassificationLoss"):
      cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=gt_classes, logits=pred_classes)
      cls_loss = obj_cells * tf.expand_dims(cls_loss, -1)
      cls_loss = 1.0 * tf.reduce_sum(cls_loss) / (num_objects + EPSILON)

    with tf.name_scope("TotalLoss"):
      total_loss = coord_loss + conf_loss + cls_loss

    if self.summary:
      tf.summary.scalar("total_loss", total_loss)
      tf.summary.scalar("regression_loss", coord_loss)
      tf.summary.scalar("object_confidence_loss", obj_conf)
      tf.summary.scalar("classification_loss", cls_loss)
      # tf.summary.scalar("AverageIOU", tf.reduce_mean((obj_cells * box_iou) / (num_objects + EPSILON)))
      # tf.summary.scalar("Recall", tf.reduce_sum(tf.to_float(box_iou > 0.5)))

    return total_loss

  def compute_iou(self, gt_boxes, pred_boxes, scope=None):
    with tf.name_scope(scope, 'IoU'):
      with tf.name_scope('Area'):
        gt_wh   = gt_boxes[..., 2:4] - gt_boxes[..., 0:2]
        pr_wh   = pred_boxes[..., 2:4] - pred_boxes[..., 0:2]
        gt_area = gt_wh[..., 0] * gt_wh[..., 1]
        pr_area = pr_wh[..., 0] * pr_wh[..., 1]

      with tf.name_scope('Intersections'):
        intersect_mins = tf.maximum(gt_boxes[..., 0:2], pred_boxes[..., 0:2])
        intersect_maxes = tf.minimum(gt_boxes[..., 2:4], pred_boxes[..., 2:4])
        intersect_hw = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersections = intersect_hw[..., 0] * intersect_hw[..., 1]
      with tf.name_scope("Unions"):
        unions = gt_area + pr_area - intersections

      iou = tf.where(tf.equal(intersections, 0.0),
                     tf.zeros_like(intersections),
                     tf.truediv(intersections, unions))

      return iou
