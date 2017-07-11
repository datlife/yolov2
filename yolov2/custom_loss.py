"""
YOLOV2 Loss Function 
"""
import keras.backend as K


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

    tx = feature_map[..., 0]
    ty = feature_map[..., 1]
    tw = feature_map[..., 2]
    th = feature_map[..., 3]
    to = feature_map[..., 4]
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