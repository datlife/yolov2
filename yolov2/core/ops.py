import tensorflow as tf


def find_and_solve_collided_indices(indices, values, shape):
    """Sometimes ground truth indices will be collided,

    shape: - a TensorShape
    We attempted to solved it by :
       * Find all a set of collided indices
       * For each set, update to the next possible index (based on iou_score)
       * If not possible, remove ground truths :(
    """
    width  = shape[0].value
    height = shape[1].value
    depth  = shape[2].value

    #  iou_scores = tf.cast(iou_scores, tf.float64)

    def reverse(index):
        # @TODO: linearize N-d dimension
        x = index / (height * depth)
        y = (index - x * height * depth) / depth
        z = index - x * height * depth - y * depth
        return tf.stack([x, y, z], -1)

    # Given a matrix N-d dimension[N_k, ...N_1, N_0], linearizing by multiply [..., N_1 * N_0, N_0, 1]
    flatten = tf.matmul(tf.cast(indices, tf.int32), [[height * depth], [depth], [1]])
    filtered_idx, idx = tf.unique(tf.squeeze(flatten))

    # @TODO: filter based on IOU
    # Convert filtered_idx to original dims
    updated_indices = tf.cast(tf.map_fn(fn=lambda i: reverse(i), elems=filtered_idx), tf.int64)
    updated_values  = tf.unsorted_segment_max(values, idx, num_segments=tf.shape(filtered_idx)[0])

    return [updated_indices, updated_values]


# The below codes are from TensorFlow team
# Citation: https://github.com/tensorflow/models/tree/master/research/object_detection

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
