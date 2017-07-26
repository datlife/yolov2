import tensorflow as tf
import keras.backend as K
from argparse import ArgumentParser

from cfg import *
from utils.parse_txt_to_inputs import parse_txt_to_inputs # Data handler for LISA dataset
from utils.data_generator import flow_from_list

from model.darknet19 import darknet19
from model.MobileYolo import MobileYolo
from model.loss import custom_loss, avg_iou, precision

K.clear_session()  # Avoid duplicate model

parser = ArgumentParser(description="Train YOLOv2")

parser.add_argument('-p', '--path', help="training data in txt format", type=str, default='training.txt')
parser.add_argument('-w', '--weights', type=str, default=None)
parser.add_argument('-e', '--epochs', help='Steps of training', type=int, default=10)
parser.add_argument('-b', '--batch', type=int, default=8)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.00000001)

args = parser.parse_args()
annotation_path = args.path
WEIGHTS_FILE = args.weights
BATCH_SIZE = args.batch


def _main_():
    # Load data
    x_train, y_train = parse_txt_to_inputs(annotation_path)

    # Build Model
    darknet = darknet19(input_size=(IMG_INPUT, IMG_INPUT, 3), pretrained_weights='yolo-coco.h5')
    yolov2 = MobileYolo(feature_extractor=darknet, num_anchors=5, num_classes=N_CLASSES, fine_grain_layer=43)
    yolov2.model.summary()

    # Construct Data Generator
    train_data_gen = flow_from_list(x_train, y_train, batch_size=BATCH_SIZE, augment_data=True)

    if WEIGHTS_FILE:
        yolov2.model.load_weights(WEIGHTS_FILE)

    yolov2.model.compile(optimizer='adam', loss=custom_loss, metrics=[avg_iou, precision])

    # Print evaluation information
    # IoU threshold
    #
    # Precision
    # Recall


def evaluate(y_true, y_pred):

    GRID_H, GRID_W = y_pred.shape[1:3]
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

    anchors_tensor = tf.to_float(K.reshape(K.variable(ANCHORS), [1, 1, 1, 5, 2]))
    netout_size = tf.to_float(K.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2]))

    box_xy = K.sigmoid(y_pred[..., :2])
    box_wh = K.exp(y_pred[..., 2:4])
    box_confidence = K.sigmoid(y_pred[..., 4:5])
    box_class_probs = K.softmax(y_pred[..., 5:])

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (box_xy + c_xy) / netout_size
    box_wh = box_wh * anchors_tensor / netout_size
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    # Y1, X1, Y2, X2
    boxes = K.concatenate([box_mins[..., 1:2], box_mins[..., 0:1], box_maxes[..., 1:2], box_maxes[..., 0:1]])

    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, -1)
    box_class_scores = K.max(box_scores, -1)
    prediction_mask = (box_class_scores >= 0.2)

    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)

    # Scale boxes back to original image shape.
    height = 960
    width = 1280

    image_dims = tf.cast(K.stack([height, width, height, width]), tf.float32)
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    nms_index = tf.image.non_max_suppression(boxes, scores, tf.Variable(10), iou_threshold=0.5)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        boxes_prediction = boxes.eval()
        scores_prediction = scores.eval()
        classes_prediction = classes.eval()

    return boxes_prediction, scores_prediction, classes_prediction


if __name__ == "__main__":
    _main_()

    # one_sample = np.tile(x_train[0:4], 8).tolist()
    # one_label  = np.tile(y_train[0:4], [8, 1])
    # print([name.split('/')[-1].split('.')[0] for name in one_sample])
    # print(one_sample[1])
    # train_data_gen = flow_from_list(one_sample, one_label, batch_size=BATCH_SIZE, augment_data=True)
    # val_data_gen = flow_from_list(one_sample, one_label,   batch_size=BATCH_SIZE, augment_data=False)
