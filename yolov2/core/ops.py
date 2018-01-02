import tensorflow as tf


def iou(boxes_list1, boxes_list2, scope=None):
    """
      Args:
        boxes_list1: Tensor holding N boxes
        boxes_list2: Tensor holding M boxes
        scope

      Returns:
        a tensor with shape [N, M] representing pairwise iou scores.
      """
    with tf.name_scope(scope, 'IOU'):
        areas1        = area(boxes_list1)
        areas2        = area(boxes_list2)
        intersections = intersection(boxes_list1, boxes_list2)

        unions = (tf.expand_dims(areas1, 1) +
                  tf.expand_dims(areas2, 0) - intersections)

    return tf.where(tf.equal(intersections, 0.0),
                    tf.zeros_like(intersections),
                    tf.truediv(intersections, unions))


def area(boxes, scope=None):
    """Computes area of boxes.

    Args:
      boxes: Tensor holding N boxes
      scope: name scope.

    Returns:
      a tensor with shape [N] representing box areas.
    """
    with tf.name_scope(scope, 'Area'):
        y_min, x_min, y_max, x_max = tf.split(value=boxes, num_or_size_splits=4, axis=-1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def intersection(boxes_list1, boxes_list2, scope=None):
    """Compute pairwise intersection areas between boxes.

    Args:
    boxlist1: Tensor holding N boxes
    boxlist2: Tensor holding M boxes
    scope: name scope.

    Returns:
    a tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope(scope, 'Intersection'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(value=boxes_list1, num_or_size_splits=4, axis=-1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(value=boxes_list2, num_or_size_splits=4, axis=-1)

        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))

        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))

        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)

    return intersect_heights * intersect_widths
