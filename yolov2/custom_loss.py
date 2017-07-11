"""
YOLOV2 Loss Function 
"""
import keras.backend as K
import numpy as np


def yolov2_loss(anchors, num_classes):
    """
    Loss function implementation in Keras
    
    :param anchors: 
    :param num_classes: 
    :return: 
    """
    def calc_loss(y_true, y_pred):
        """
        :param y_true: 
        :param y_pred: output from YOLOv2 - a conv layer
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
        :return: 
        """

        pred_bx, pred_by, pred_bw, pred_bh, prob_obj, prob_cls = _transform_conv_layer_to_prediction(anchors, y_pred)

        # Transform ground truth boxes into feature_map size
        gt_bx = 0
        gt_by = 0
        gt_bw = 0
        gt_bh = 0
        gt_cls = 0

        # Calculate Loss

        return 0.0

    return calc_loss


def _transform_conv_layer_to_prediction(anchors, y_pred):
    """

    :param anchors: 
    :param y_pred: 
    :return: 
    """
    num_anchors = len(anchors)
    # Reshape anchors to [batch, height, width, num_anchors, box_params]
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_shape = K.shape(y_pred)[1:3]  # assuming channels last
    height_index = K.arange(0, stop=conv_shape[0])
    height_index = K.tile(height_index, [conv_shape[1]])

    width_index = K.arange(0, stop=conv_shape[1])
    width_index = K.tile(K.expand_dims(width_index, 0), [conv_shape[0], 1])
    width_index = K.flatten(K.transpose(width_index))

    # Create index for cell map
    conv_index = K.transpose(K.stack([height_index, width_index]))
    conv_index = K.reshape(conv_index, [1, conv_shape[0], conv_shape[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(y_pred))

    feature_map = K.reshape(y_pred, [-1, conv_shape[0], conv_shape[1], num_anchors, num_classes + 5])
    conv_shape = K.cast(K.reshape(conv_shape, [1, 1, 1, 1, 2]), K.dtype(feature_map))

    tx    = feature_map[..., 0]
    ty    = feature_map[..., 1]
    tw    = feature_map[..., 2]
    th    = feature_map[..., 3]
    to    = feature_map[..., 4]
    probs = feature_map[..., 5:]

    cx = conv_index[..., 0]
    cy = conv_index[..., 1]
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
