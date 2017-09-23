"""
YOLOv2 Meta-Architecture

        images --> feature extractor --> feature map --> detector --> output feature map

In this file, we define three different detector:
   * Original YOLOv2 Detector
   * MobileNet-type detector
   * DenseNet-type detector

-----------------------
Example usage: This code will define pretrained YOLOv2 on COCO Dataset (80 classes)

from models.YOLOv2 import YOLOv2
from models.loss import custom_loss
from models.FeatureExtractor import FeatureExtractor

# Define YOLOv2 with Darknet-19 as feature extractor and use 'yolov2' as detector
darknet = FeatureExtractor(img_size=(608, 608, 3), model='yolov2')

yolo    = YOLOv2(num_classes       = 80,
                 anchors           = np.array(ANCHORS),
                 feature_extractor = darknet,
                 detector          = 'yolov2')

# Compile model
model = yolo.model.compile('adam',loss=custom_loss)

"""
import numpy as np
import tensorflow as tf
import keras.backend as K

from zoo.darknet19 import conv_block
from zoo.mobilenet import _depthwise_conv_block, relu6

from keras.models import Model
from keras.regularizers import l2
from keras.layers.merge import concatenate
from keras.layers import Lambda, Conv2D, BatchNormalization, Activation

from softmaxtree.Tree import SoftMaxTree

FINE_GRAINED_LAYERS = {'yolov2': 'leaky_re_lu_13',
                       'mobilenet': 'conv_pw_11_relu',
                       'densenet': 'leaky_re_lu_136'}


def yolov2_detector(feature_extractor, num_classes, num_anchors, fine_grained_layers):
    """
    Original YOLOv2 Implementation
    :param feature_extractor:
    :param num_classes:
    :param num_anchors:

    :return:
    """

    inputs = feature_extractor.model.output
    i = feature_extractor.model.get_layer(fine_grained_layers).output

    x = conv_block(inputs, 1024, (3, 3))
    x = conv_block(x, 1024, (3, 3))
    x2 = x

    # Reroute
    x = conv_block(i, 64, (1, 1))
    x = Lambda(lambda x: tf.space_to_depth(x, block_size=2),
               lambda shape: [shape[0], shape[1] / 2, shape[2] / 2, 2 * 2 * shape[-1]] if shape[1] else
               [shape[0], None, None, 2 * 2 * shape[-1]],
               name='space_to_depth_x2')(x)

    x = concatenate([x, x2])
    x = conv_block(x, 1024, (3, 3))
    x = Conv2D(num_anchors * (num_classes + 5), (1, 1), name='yolov2', kernel_regularizer=l2(5e-4))(x)

    return x


def mobile_detector(feature_extractor, num_classes, num_anchors, fine_grained_layers):
    """
    Mobile Detector Implementation
    :param feature_extractor:
    :param num_classes:
    :param num_anchors:
    :param fine_grained_layers

    :return:
    """
    inputs = feature_extractor.model.output
    i = feature_extractor.model.get_layer(fine_grained_layers).output

    x = _depthwise_conv_block(inputs, 1024, 1.0, block_id=14)
    x = _depthwise_conv_block(x, 1024, 1.0, block_id=15)
    x2 = x

    # Reroute
    x = Conv2D(64, (1, 1), padding='same', use_bias=False, strides=(1, 1))(i)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    x = Lambda(lambda x: tf.space_to_depth(x, block_size=2),
               lambda shape: [shape[0], shape[1] / 2, shape[2] / 2, 2 * 2 * shape[-1]] if shape[1] else
               [shape[0], None, None, 2 * 2 * shape[-1]],
               name='space_to_depth_x2')(x)
    x = concatenate([x, x2])

    x = _depthwise_conv_block(x, 1024, 1.0, block_id=16)
    x = Conv2D(num_anchors * (num_classes + 5), (1, 1))(x)

    return x


def densenet_detector(feature_extractor, num_classes, num_anchors, fine_grained_layers):
    """
    DenseNet Implementation - For Now I keeped this architecture the same as YOLov2 due to a few incompatibilities
    :param feature_extractor:
    :param num_classes:
    :param num_anchors:
    :param fine_grained_layers

    :return:
    """
    inputs = feature_extractor.model.output
    i = feature_extractor.model.get_layer(fine_grained_layers).output

    x = conv_block(inputs, 1024, (3, 3), name='detector_1')
    x = conv_block(x, 1024, (3, 3), name='detector_2')
    x2 = x

    # Reroute
    x = conv_block(i, 64, (1, 1), name='detector_3')
    x = Lambda(lambda x: tf.space_to_depth(x, block_size=2),
               lambda shape: [shape[0], shape[1] / 2, shape[2] / 2, 2 * 2 * shape[-1]] if shape[1] else
               [shape[0], None, None, 2 * 2 * shape[-1]],
               name='space_to_depth_x2')(x)

    x = concatenate([x, x2], name='concat_2')
    x = conv_block(x, 1024, (3, 3), name='detector_4')
    x = Conv2D(num_anchors * (num_classes + 5), (1, 1), name='yolov2-densenet')(x)

    return x


DETECTOR = {'yolov2': yolov2_detector,
            'mobilenet': mobile_detector,
            'densenet': densenet_detector}


class YOLOv2(object):
    def __init__(self,
                 is_training,
                 num_classes,
                 anchors,
                 feature_extractor,
                 detector,
                 use_hierarchical_tree=False,
                 tree_path=None
                 ):
        """
        YOLOv2 Meta-Architecture

        images --> feature extractor --> feature map --> detector --> output feature map

        :param num_classes:
        :param anchors:
        :param is_training:
        :param feature_extractor:
        :param detector:
        """

        self.num_classes       = num_classes
        self.anchors           = anchors
        self._is_training      = is_training
        self.feature_extractor = feature_extractor

        self.fine_grained_layers = FINE_GRAINED_LAYERS[feature_extractor.name]
        self.detector = DETECTOR[detector](feature_extractor, num_classes, len(anchors), self.fine_grained_layers)

        # YOLOv2 Model
        self.model = Model(inputs=feature_extractor.model.input,
                           outputs=self.detector)

        # Setting Tree
        self.use_tree = use_hierarchical_tree

        if self.use_tree is True:
            if tree_path is None:
                raise ValueError("Invalid Tree Path. Please check cfg.py if tree was used.")
            self.softmax_tree = SoftMaxTree(tree_file=tree_path)

    def post_process(self, img_shape, n_classes=80, iou_threshold=0.5, score_threshold=0.6, mode=2):
        """
        Input: out feature map from network

        Output:
           Bounding Boxes - Classes - Probabilities

        """

        N_ANCHORS = len(self.anchors)
        ANCHORS = self.anchors

        prediction = self.model.output
        pred_shape = tf.shape(prediction)
        GRID_H, GRID_W = pred_shape[1], pred_shape[2]

        prediction = K.reshape(prediction, [-1, pred_shape[1], pred_shape[2], N_ANCHORS, n_classes + 5])

        # Create off set map
        c_xy = self._create_offset_map(K.shape(pred_shape))

        anchors_tensor  = tf.to_float(K.reshape(ANCHORS, [1, 1, 1, N_ANCHORS, 2]))
        netout_size     = tf.to_float(K.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]))

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

        if self.use_tree is False:
            box_scores = box_confidence * K.softmax(box_class_probs)
            box_classes = K.argmax(box_scores, -1)
        else:
            n_children = len(self.softmax_tree.tree_dict[0].children)
            child_idx = self.softmax_tree.tree_dict[0].children[0].id

            if mode == 0:
                box_scores = box_confidence
                box_classes = K.argmax(box_scores, -1)

            if mode == 1:
                box_scores = box_confidence * K.softmax(box_class_probs[..., 0:0 + n_children])
                box_classes = K.argmax(box_scores, -1) + child_idx

            if mode == 2:
                scores = []
                abst_scores = box_confidence * K.softmax(box_class_probs[..., 0:0 + n_children])

                for i, node in enumerate(self.softmax_tree.tree_dict[0].children):
                    node_prob = abst_scores[..., i: i + 1]
                    if node.children:
                        id = node.children[0].id
                        sub_probs = node_prob * K.softmax(box_class_probs[..., id:id + len(node.children)])
                    else:
                        sub_probs = node_prob
                    scores.append(sub_probs)

                box_scores = tf.concat(scores, 4)
                box_classes = K.argmax(box_scores, -1) + len(self.softmax_tree.tree_dict[0].children) + 2

        box_class_scores = K.max(box_scores, -1)
        prediction_mask = (box_class_scores >= score_threshold)

        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)

        # Scale boxes back to original image shape.
        height, width = img_shape[0], img_shape[1]
        image_dims = tf.cast(K.stack([height, width, height, width]), tf.float32)
        image_dims = K.reshape(image_dims, [1, 4])
        boxes = boxes * image_dims

        nms_index = tf.image.non_max_suppression(boxes, scores, 10, iou_threshold)
        boxes     = tf.gather(boxes, nms_index)
        scores    = tf.gather(scores, nms_index)
        classes   = tf.gather(classes, nms_index)

        return boxes, classes, scores

    def loss_func(self, y_true, y_pred):
        """
        YOLOv2 Loss Function Implementation

        Input: out feature map from network

        Output:
           A scalar - loss value for back propagation

        ------------------

        Loss of YOLOv2 Implementation. Few Notes
            * What we get from the CNN is a feature map (imagine as a 3-D box)
            * Each cell in a feature map is a vector size N_ANCHORS* (5 + N_CLASSES)  as:

                ------------- ANCHOR 1---------------  -------- ANCHORS 2 -------------  ...... ------------ANCHOR N -----------
                [tx, ty, tw, th, to , label_vector..],[tx1, ty1, tw1, th1, label_vector]......[tx_n, ty_n, tw_n, th_m, label...]
                ----------------------------------------------------------------------------------------------------------------
                                                         One cell in a feature map


                    * tx, ty : predicts of relative center of bounding box to its current cell. Therefore, true center points of
                               a prediction would be :
                                    xc = sigmoid(tx) + cx
                                    yc = sigmoid(ty) + cy
                    * tw, th: predicts the scaling value for true width and height of the bounding box based on the anchor as:
                                    w   = exp(tw) * px
                                    h   = exp(th) * py
                    * to :     objectiveness of the cell : the probability of having an object in the cell
                    * label:   classification vector to calculate soft-max
        """
        N_ANCHORS = len(self.anchors)
        N_CLASSES = self.num_classes

        pred_shape = K.shape(y_pred)[1:3]
        gt_shape   = K.shape(y_true)  # shape of ground truth value
        GRID_H     = tf.cast(pred_shape[0], tf.int32)  # shape of output feature map
        GRID_W     = tf.cast(pred_shape[1], tf.int32)

        output_size = tf.cast(tf.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]), tf.float32)
        y_pred = tf.reshape(y_pred, [-1, pred_shape[0], pred_shape[1], N_ANCHORS, N_CLASSES + 5])
        y_true = tf.reshape(y_true, [-1, gt_shape[1], gt_shape[2], N_ANCHORS, N_CLASSES + 5])

        # Grid Map to calculate offset
        c_xy = self._create_offset_map(K.shape(y_pred))

        # Scale anchors to correct aspect ratio
        pred_box_xy   = (tf.sigmoid(y_pred[:, :, :, :, :2]) + c_xy) / output_size
        pred_box_wh   = tf.exp(y_pred[:, :, :, :, 2:4]) * np.reshape(self.anchors, [1, 1, 1, N_ANCHORS, 2]) / output_size
        pred_box_wh   = tf.sqrt(pred_box_wh)
        pred_box_conf = tf.sigmoid(y_pred[:, :, :, :, 4:5])
        pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])

        # Adjust ground truth
        true_box_xy = y_true[:, :, :, :, 0:2]
        true_box_wh = tf.sqrt(y_true[:, :, :, :, 2:4])

        # adjust confidence
        pred_tem_wh   = tf.pow(pred_box_wh, 2) * output_size
        pred_box_ul   = pred_box_xy - 0.5 * pred_tem_wh
        pred_box_bd   = pred_box_xy + 0.5 * pred_tem_wh
        pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]

        true_tem_wh   = tf.pow(true_box_wh, 2) * output_size
        true_box_ul   = true_box_xy - 0.5 * true_tem_wh
        true_box_bd   = true_box_xy + 0.5 * true_tem_wh
        true_box_area = true_tem_wh[:, :, :, :, 0] * true_tem_wh[:, :, :, :, 1]

        intersect_ul   = tf.maximum(pred_box_ul, true_box_ul)
        intersect_br   = tf.minimum(pred_box_bd, true_box_bd)
        intersect_wh   = tf.maximum(intersect_br - intersect_ul, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
        best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
        best_box = tf.to_float(best_box)
        true_box_conf = tf.expand_dims(best_box * y_true[:, :, :, :, 4], -1)
        true_box_prob = y_true[:, :, :, :, 5:]

        # Localization Loss
        weight_coor = 5.0 * tf.concat(4 * [true_box_conf], 4)
        true_boxes = tf.concat([true_box_xy, true_box_wh], 4)
        pred_boxes = tf.concat([pred_box_xy, pred_box_wh], 4)
        loc_loss = tf.pow(true_boxes - pred_boxes, 2) * weight_coor
        loc_loss = tf.reshape(loc_loss, [-1, tf.cast(GRID_W * GRID_H, tf.int32) * N_ANCHORS * 4])
        loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, 1))

        # NOTE: YOLOv2 does not use cross-entropy loss.
        if self.use_tree is False:
            # Object Confidence Loss
            weight_conf = 0.5 * (1. - true_box_conf) + 5.0 * true_box_conf
            obj_conf_loss = tf.pow(true_box_conf - pred_box_conf, 2) * weight_conf
            obj_conf_loss = tf.reshape(obj_conf_loss, [-1, tf.cast(GRID_W * GRID_H, tf.int32) * N_ANCHORS])
            obj_conf_loss = tf.reduce_mean(tf.reduce_sum(obj_conf_loss, 1))

            # Category Loss
            weight_prob = 1.0 * tf.concat(N_CLASSES * [true_box_conf], 4)
            category_loss = tf.pow(true_box_prob - pred_box_prob, 2) * weight_prob
            category_loss = tf.reshape(category_loss, [-1, tf.cast(GRID_W * GRID_H, tf.int32) * N_ANCHORS * N_CLASSES])
            category_loss = tf.reduce_mean(tf.reduce_sum(category_loss, 1))

            loss = 0.5 * (loc_loss + obj_conf_loss + category_loss)

        else:
            category_loss = self._calculate_softmax(idx=0,
                                                    logits=y_pred[..., 4:],
                                                    labels=y_true[..., 4:],
                                                    pred_obj_conf=pred_box_conf,
                                                    true_obj_conf=true_box_conf)

            # category_loss = tf.Print(category_loss, [category_loss])
            loss = 0.5 * (loc_loss + category_loss)

        return loss

    def _create_offset_map(self, output_shape):
        """
        In Yolo9000 paper, Grid map to calculate offsets for each cell in the output feature map
        """
        GRID_H = tf.cast(output_shape[1], tf.int32)  # shape of output feature map
        GRID_W = tf.cast(output_shape[2], tf.int32)

        cx = tf.cast((K.arange(0, stop=GRID_W)), dtype=tf.float32)
        cx = K.tile(cx, [GRID_H])
        cx = K.reshape(cx, [-1, GRID_H, GRID_W, 1])

        cy = K.cast((K.arange(0, stop=GRID_H)), dtype=tf.float32)
        cy = K.reshape(cy, [-1, 1])
        cy = K.tile(cy, [1, GRID_W])
        cy = K.reshape(cy, [-1])
        cy = K.reshape(cy, [-1, GRID_H, GRID_W, 1])

        c_xy = tf.stack([cx, cy], -1)
        c_xy = K.cast(c_xy, tf.float32)

        return c_xy

    def _calculate_softmax(self, idx, logits, labels, pred_obj_conf, true_obj_conf):
        """
        Update Probabilities of each labels accordingly to Hierarchical Structure
        :param idx:   default = -1 / starting from root of soft-max tree
        :param logits:
        :param labels:
        :return:
        """
        loss = 0.0
        gt_shape = tf.shape(labels)[1:3]
        GRID_W, GRID_H = gt_shape[0], gt_shape[1]

        if idx == 0:
            # Object Confidence Loss
            weight_conf = 0.5 * (1. - true_obj_conf) + 5.0 * true_obj_conf
            loss = tf.pow(true_obj_conf - pred_obj_conf, 2) * weight_conf
            loss = tf.reshape(loss, [-1, tf.cast(GRID_W * GRID_H, tf.int32) * len(self.anchors)])
            loss = tf.reduce_mean(tf.reduce_sum(loss, 1))

        if self.softmax_tree[idx].children:  # traverse until reaching the leaf
            weights_prob = tf.concat(len(self.softmax_tree[idx].children) * [true_obj_conf], 4)
            first_child = self.softmax_tree[idx].children[0].id

            sub_logits = logits[..., first_child:first_child + len(self.softmax_tree[idx].children)]
            sub_label  = labels[..., first_child:first_child + len(self.softmax_tree[idx].children)]

            # # Calculate Parent Probability
            if idx == 0:
                sub_logits = pred_obj_conf * tf.nn.softmax(sub_logits)
            else:
                parent = self.softmax_tree[idx].parent.children[0].id
                parent_softmax = tf.nn.softmax(logits[..., parent: parent + len(self.softmax_tree[idx].parent.children)])
                parent_softmax = pred_obj_conf * parent_softmax

                for i, child in enumerate(self.softmax_tree[idx].parent.children):
                    if child.id == idx:
                        parent_prob = parent_softmax[..., i:i + 1]
                        sub_logits = parent_prob * tf.nn.softmax(sub_logits)
                        break

            sub_loss = tf.pow(sub_label - sub_logits, 2) * weights_prob
            sub_loss = tf.reshape(sub_loss, [-1, GRID_H * GRID_W * len(self.anchors) * (len(self.softmax_tree[idx].children))])
            sub_loss = tf.reduce_mean(tf.reduce_sum(sub_loss, 1))
            loss += sub_loss

            # Calculate loss of each children
            for children in self.softmax_tree[idx].children:
                sub_loss = self._calculate_softmax(idx=children.id,
                                                   logits=logits,
                                                   labels=labels,
                                                   pred_obj_conf=pred_obj_conf,
                                                   true_obj_conf=true_obj_conf)
                loss += sub_loss
        return loss


def reroute(x1, x2, stride=2):
    x = conv_block(x1, 64, (1, 1))
    x = Lambda(lambda x: tf.space_to_depth(x, block_size=stride),
               lambda shape: [shape[0], shape[1] / stride, shape[2] / stride, stride * stride * shape[-1]] if shape[
                   1] else
               [shape[0], None, None, stride * stride * shape[-1]],
               name='space_to_depth_x2')(x)
    x = concatenate([x, x2])

    return x
