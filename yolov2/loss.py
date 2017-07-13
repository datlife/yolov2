"""
YOLOV2 Loss Function 
"""
import keras.backend as K
import numpy as np
import tensorflow as tf


def yolov2_loss(anchors, num_classes):
    """
    Loss function implementation in Keras
    """
    def calc_loss(y_true, y_pred):
        n_anchors = len(anchors)

        prediction   = _output_to_prediction(y_pred,  anchors, num_classes)
        gt_boxes     = _extract_ground_truth(y_true,  anchors, num_classes)

        pred_bx, pred_by, pred_bw, pred_bh, prob_obj, prob_cls = prediction

        # Transform ground truth boxes into feature_map size
        gt_bx  = gt_boxes[..., 0]
        gt_by  = gt_boxes[..., 1]
        gt_bw  = gt_boxes[..., 2]
        gt_bh  = gt_boxes[..., 3]
        gt_cls = gt_boxes[..., 4]

        # Calculate IoU between prediction and ground truth (for each grid cell in output layer)

        # Find the best IOU > certain threshold (0.6)

        #

        # Calculate Loss

        return 0

    return calc_loss


def _extract_ground_truth(y_true, anchors, num_classes):
    """
    y_true shape is [GT_BOXES, DETECTORS_MASK, MASK_AGAIN]
        GT_BOXES        = [batch_size, xc, yc, w, h, gt_class]
    :param y_true: 
    :return: 
    """
    gt_shape = K.shape(y_true)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(y_true.eval().shape)
        print(y_true.eval())
        # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    ground_truth  = K.reshape(y_true, [gt_shape[0], 1, 1, len(anchors), gt_shape[1], gt_shape[2]])
    return ground_truth


def _output_to_prediction(y_pred, anchors, num_classes):
    """
            According to YOLO900 paper, at each cell of output conv layer, the network would predict 5 bounding boxes. In
            each of these bounding boxes,  the network predicts: [tx, ty, tw, th, to].
                
            For example: 
                A cell i is offset from top left a distance (cx, cy) and the anchor has (pw, ph). 
                The prediction is:
                                    bx = sigmoid(tx) + cx
                                    by = sigmoid(ty) + cy
                                    bw = pw*exp(tw)
                                    bh = ph*exp(th)
                Pr(obj)*IOU(box, obj)  = sigmoid(to)   
    :param anchors: 
    :param y_pred: 
    :return: 
    """
    num_anchors = len(anchors)
    # Reshape anchors to [batch, height, width, num_anchors, box_params]
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_shape   = K.shape(y_pred)[1:3]  # assuming 'channels_last'
    feature_map = K.reshape(y_pred, [-1, conv_shape[0], conv_shape[1], num_anchors, num_classes + 5])
    conv_shape  = K.cast(K.reshape(conv_shape, [1, 1, 1, 1, 2]), K.dtype(feature_map))

    tx    = feature_map[..., 0]
    ty    = feature_map[..., 1]
    tw    = feature_map[..., 2]
    th    = feature_map[..., 3]
    to    = feature_map[..., 4]
    probs = feature_map[..., 5:]

    cx = 0
    cy = 0
    px = anchors_tensor[..., 0]
    py = anchors_tensor[..., 1]

    pred_bx  = K.sigmoid(tx) + cx
    pred_by  = K.sigmoid(ty) + cy
    pred_bw  = px * K.exp(tw)
    pred_bh  = py * K.exp(th)
    prob_obj = K.sigmoid(to)
    prob_cls = K.softmax(probs)

    # Adjust predictions to each spatial grid point and anchor size.
    # pred_box_xy = (pred_box_xy + conv_index) / conv_shape    # (bx = sigmoid(tx) + cx)
    # pred_box_wh = pred_box_wh * anchors_tensor / conv_shape  # (bw = pw*exp(to))

    return pred_bx, pred_by, pred_bw, pred_bh, prob_obj, prob_cls


def pre_process_true_boxes(true_boxes, anchors, image_size):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative [x, y, w, h, class].
        **Relative coordinates** are in the range [0, 1] indicating a percentage
        of the original image dimensions.
        
    anchors : array
        List of anchors in form of [w, h].
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
        
    image_size : array-like
        List of image dimensions in form of [h, w, c] in pixels.

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
        
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
        
    """
    height, width, _ = image_size
    num_anchors = len(anchors)

    # Down-sampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hard-coding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width  % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'

    conv_height = height // 32
    conv_width  = width // 32

    num_box_params      = true_boxes.shape[1]
    detectors_mask      = np.zeros((conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros((conv_height, conv_width, num_anchors, num_box_params), dtype=np.float32)

    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5]
        box = box[0:4] * np.array([conv_width, conv_height, conv_width, conv_height])
        i = np.floor(box[1]).astype('int')  # list of y_center
        j = np.floor(box[0]).astype('int')  # list of x_center

        best_iou = 0
        best_anchor = 0
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        if best_iou > 0:
            detectors_mask[i, j, best_anchor] = 1
            adjusted_box = np.array([box[0] - j,
                                     box[1] - i,
                                     np.log(box[2] / anchors[best_anchor][0]),
                                     np.log(box[3] / anchors[best_anchor][1]),
                                     box_class],
                                    dtype=np.float32)
            matching_true_boxes[i, j, best_anchor] = adjusted_box

    return detectors_mask, matching_true_boxes


def yolo_loss(anchors, n_classes):
    def custom_loss(y_true, y_pred):
        #         CONV_SHAPE = tf.cast(K.shape(y_pred)[1:3], dtype=tf.int32)

        GRID_W, GRID_H = 40, 30
        #         GRID_W, GRID_H = CONV_SHAPE[0], CONV_SHAPE[1]  # a tiny hack to get width, height of output layer
        NORM_W, NORM_H = GRID_W * 32, GRID_H * 32  # Scale back to get image input size

        SCALE_NOOB, SCALE_CONF, SCALE_COOR, SCALE_PROB = 0.5, 5.0, 5.0, 1.0

        y_pred = K.reshape(y_pred, [-1, GRID_W, GRID_H, len(anchors), n_classes + 5])
        # Adjust prediction  : bx = sigmoid(tx) + cx
        pred_box_xy = tf.sigmoid(y_pred[:, :, :, :, :2])

        # adjust w and h : bw = pw*exp(tw)
        pred_box_wh = tf.exp(y_pred[:, :, :, :, 2:4]) * np.reshape(anchors, [1, 1, 1, len(anchors), 2])

        # scaling wh relative to top-left corner of feature map
        pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(GRID_W), float(GRID_H)], [1, 1, 1, 1, 2]))
        pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)  # adjust confidence
        pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])  # adjust probability

        y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)

        # Adjust ground truth
        # adjust x and y
        gt_dim = K.shape(y_true)
        y_true = K.reshape(y_true, [gt_dim[0], GRID_W, GRID_H, gt_dim[1], gt_dim[2]])
        center_xy = .5 * (y_true[:, :, :, :, 0:2] + y_true[:, :, :, :, 2:4])
        center_xy = center_xy / np.reshape([(float(NORM_W) / GRID_W), (float(NORM_H) / GRID_H)], [1, 1, 1, 1, 2])
        true_box_xy = center_xy - tf.floor(center_xy)

        # adjust w and h
        true_box_wh = (y_true[:, :, :, :, 2:4] - y_true[:, :, :, :, 0:2])
        true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(NORM_W), float(NORM_H)], [1, 1, 1, 1, 2]))

        # adjust confidence
        pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2])
        pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
        pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
        pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh

        true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2])
        true_box_area = true_tem_wh[:, :, :, :, 0] * true_tem_wh[:, :, :, :, 1]
        true_box_ul = true_box_xy - 0.5 * true_tem_wh  # upper left
        true_box_br = true_box_xy + 0.5 * true_tem_wh  # lower right

        # Find the IOU between GT and prediction
        intersect_ul = tf.maximum(pred_box_ul, true_box_ul)
        intersect_br = tf.minimum(pred_box_bd, true_box_br)
        intersect_wh = intersect_br - intersect_ul
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect_area = intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

        iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
        best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
        best_box = tf.to_float(best_box)
        true_box_conf = tf.expand_dims(best_box * y_true[:, :, :, :, 4], -1)

        # adjust confidence
        true_box_prob = y_true[:, :, :, :, 5:]
        y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
        # y_true = tf.Print(y_true, [true_box_wh], message='DEBUG', summarize=30000)

        # ## Compute the weights
        weight_coor = tf.concat(4 * [true_box_conf], 4)
        weight_coor = SCALE_COOR * weight_coor
        weight_conf = SCALE_NOOB * (1. - true_box_conf) + SCALE_CONF * true_box_conf
        weight_prob = tf.concat(n_classes * [true_box_conf], 4)
        weight_prob = SCALE_PROB * weight_prob
        weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)

        # ## Finalize the loss
        loss = tf.pow(y_pred - y_true, 2)
        loss = loss * weight
        loss = tf.reshape(loss, [-1, GRID_W * GRID_H * len(anchors) * (4 + 1 + n_classes)])
        loss = tf.reduce_sum(loss, 1)
        loss = .5 * tf.reduce_mean(loss)

        return loss

    return custom_loss
