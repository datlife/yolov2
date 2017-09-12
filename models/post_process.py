import keras.backend as K
import tensorflow as tf
from cfg import ENABLE_TREE, TREE_FILE


def post_process(yolov2, img_shape, n_classes=80, anchors=None, iou_threshold=0.5, score_threshold=0.6, mode=2):
    if ENABLE_TREE is True:
        from softmaxtree.Tree import SoftMaxTree
        softmax_tree = SoftMaxTree(TREE_FILE)

    N_ANCHORS  = len(anchors)
    ANCHORS    = anchors

    prediction = yolov2.output
    pred_shape = tf.shape(prediction)
    GRID_H, GRID_W = pred_shape[1], pred_shape[2]

    prediction = K.reshape(prediction, [-1, pred_shape[1], pred_shape[2], N_ANCHORS, n_classes + 5])

    cx = tf.cast((K.arange(0, stop=GRID_W)), dtype=tf.float32)
    cx = K.tile(cx, [GRID_H])
    cx = K.reshape(cx, [-1, GRID_H, GRID_W, 1])

    cy = K.cast((K.arange(0, stop=GRID_H)), dtype=tf.float32)
    cy = K.reshape(cy, [-1, 1])
    cy = K.tile(cy, [1, GRID_W])
    cy = K.reshape(cy, [-1])
    cy = K.reshape(cy, [-1, GRID_H, GRID_W, 1])

    c_xy = tf.stack([cx, cy], -1)
    c_xy = tf.to_float(c_xy)

    anchors_tensor = tf.to_float(K.reshape(ANCHORS, [1, 1, 1, N_ANCHORS, 2]))
    netout_size = tf.to_float(K.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]))

    box_xy          = K.sigmoid(prediction[..., :2])
    box_wh          = K.exp(prediction[..., 2:4])
    box_confidence  = K.sigmoid(prediction[..., 4:5])
    box_class_probs = prediction[..., 5:]

    # Shift center points to its grid cell accordingly (Ref: YOLO-9000 loss function)
    box_xy    = (box_xy + c_xy) / netout_size
    box_wh    = (box_wh * anchors_tensor) / netout_size
    box_mins  = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    # Y1, X1, Y2, X2
    boxes = K.concatenate([box_mins[..., 1:2], box_mins[..., 0:1],  # Y1 X1
                           box_maxes[..., 1:2], box_maxes[..., 0:1]])  # Y2 X2

    if ENABLE_TREE is False:
        box_scores = box_confidence * K.softmax(box_class_probs)
        box_classes = K.argmax(box_scores, -1)
    else:
        n_children = len(softmax_tree.tree_dict[0].children)
        child_idx = softmax_tree.tree_dict[0].children[0].id

        if mode == 0:
            box_scores = box_confidence
            box_classes = K.argmax(box_scores, -1)

        if mode == 1:
            box_scores = box_confidence * K.softmax(box_class_probs[..., 0:0 + n_children])
            box_classes = K.argmax(box_scores, -1) + child_idx

        if mode == 2:
            scores = []
            abst_scores = box_confidence * K.softmax(box_class_probs[..., 0:0 + n_children])

            for i, node in enumerate(softmax_tree.tree_dict[0].children):
                node_prob = abst_scores[..., i: i + 1]
                if node.children:
                    id = node.children[0].id
                    sub_probs = node_prob * K.softmax(box_class_probs[..., id:id + len(node.children)])
                else:
                    sub_probs = node_prob
                scores.append(sub_probs)

            box_scores = tf.concat(scores, 4)
            box_classes = K.argmax(box_scores, -1) + len(softmax_tree.tree_dict[0].children) + 2

    box_class_scores = K.max(box_scores, -1)
    prediction_mask = (box_class_scores >= score_threshold)

    boxes   = tf.boolean_mask(boxes, prediction_mask)
    scores  = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)

    # Scale boxes back to original image shape.
    height, width = img_shape[0], img_shape[1]
    image_dims = tf.cast(K.stack([height, width, height, width]), tf.float32)
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    nms_index = tf.image.non_max_suppression(boxes, scores, 10, iou_threshold)
    boxes = tf.gather(boxes, nms_index)
    scores = tf.gather(scores, nms_index)
    classes = tf.gather(classes, nms_index)

    return boxes, classes, scores
