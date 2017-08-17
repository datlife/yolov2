import numpy as np
import keras.backend as K
import tensorflow as tf


def predict(model, img, n_classes=80, anchors=None, iou_threshold=0.5, score_threshold=0.6):

    input_img = np.expand_dims(img, 0)

    # Making prediction
    with K.get_session().as_default():
        prediction = model.predict(input_img)

    pred_shape = np.shape(prediction)
    N_ANCHORS  = len(anchors)
    ANCHORS    = anchors

    prediction = np.reshape(prediction, [-1, pred_shape[1], pred_shape[2], N_ANCHORS, n_classes + 5])
    GRID_H, GRID_W = prediction.shape[1:3]

    # Create GRID-cell map
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

    anchors_tensor = tf.to_float(K.reshape(K.variable(ANCHORS), [1, 1, 1, N_ANCHORS, 2]))
    netout_size = tf.to_float(K.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]))

    box_xy          = K.sigmoid(prediction[..., :2])
    box_wh          = K.exp(prediction[..., 2:4])
    box_confidence  = K.sigmoid(prediction[..., 4:5])
    box_class_probs = K.softmax(prediction[..., 5:])

    # Shift center points to its grid cell accordingly (Ref: YOLO-9000 loss function)
    box_xy    = (box_xy + c_xy) / netout_size
    box_wh    = (box_wh * anchors_tensor) / netout_size
    box_mins  = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    # Y1, X1, Y2, X2
    boxes = K.concatenate([box_mins[..., 1:2], box_mins[..., 0:1], box_maxes[..., 1:2], box_maxes[..., 0:1]])

    box_scores = box_confidence * box_class_probs

    box_classes = K.argmax(box_scores, -1)
    box_class_scores = K.max(box_scores, -1)
    prediction_mask = (box_class_scores >= score_threshold)

    boxes   = tf.boolean_mask(boxes, prediction_mask)
    scores  = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)

    # Scale boxes back to original image shape.
    height, width, _ = img.shape

    image_dims = tf.cast(K.stack([height, width, height, width]), tf.float32)
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    nms_index = tf.image.non_max_suppression(boxes, scores, tf.Variable(10), iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)

    init = tf.local_variables_initializer()
    K.get_session().run(init)
    boxes_prediction = boxes.eval()
    scores_prediction = scores.eval()
    classes_prediction = classes.eval()

    return boxes_prediction, classes_prediction, scores_prediction
