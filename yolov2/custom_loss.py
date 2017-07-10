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
        :param y_pred: 
            According to YOLO900 paper, at each cell in the last layer, the network would predict 5 bounding boxes. For
            each bounding box,  the network predict: [tx, ty, tw, th, to].
                
            
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
        num_anchors = len(anchors)
        # Reshape anchors to [batch, height, width, num_anchors, box_params]
        anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

        # Dynamic implementation of conv dims for fully convolutional model.
        conv_shape = K.shape(y_pred)[1:3]  # assuming channels last
        height_index = K.arange(0, stop=conv_shape[0])
        height_index = K.tile(height_index, [conv_shape[1]])

        weight_index = K.arange(0, stop=conv_shape[1])
        weight_index = K.tile(K.expand_dims(weight_index, 0), [conv_shape[0], 1])
        weight_index = K.flatten(K.transpose(weight_index))

        # Create index for cell map
        conv_index = K.transpose(K.stack([height_index, weight_index]))
        conv_index = K.reshape(conv_index, [1, conv_shape[0], conv_shape[1], 1, 2])
        conv_index = K.cast(conv_index, K.dtype(y_pred))

        feature_map = K.reshape(y_pred, [-1, conv_shape[0], conv_shape[1], num_anchors, num_classes + 5])
        conv_shape   = K.cast(K.reshape(conv_shape, [1, 1, 1, 1, 2]), K.dtype(feature_map))

        pred_box_xy          = K.sigmoid(feature_map[..., :2])
        pred_box_wh          = K.exp(feature_map[..., 2:4])
        pred_box_confidence   = K.sigmoid(feature_map[..., 4:5])
        pred_box_class_probs = K.softmax(feature_map[..., 5:])

        # Adjust predictions to each spatial grid point and anchor size.
        pred_box_xy = (pred_box_xy + conv_index) / conv_shape    # (bx = sigmoid(tx) + cx)
        pred_box_wh = pred_box_wh * anchors_tensor / conv_shape  # (bw = pw*exp(to))


        return 0.0

    return calc_loss

