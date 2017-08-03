import cv2
import os
import fnmatch

import tensorflow as tf
import keras.backend as K

from model.darknet19 import darknet19
from model.YOLOv2 import YOLOv2
from model.MobileYolo import MobileYolo

from cfg import *
from utils.draw_boxes import Box, draw
K.clear_session()  # to avoid duplicating model

WEIGHTS_PATH = './yolov2.weights'
MODE         = 1
IOU_THRESH   = 0.5
SCORE_THRESH = 0.5
# Map int to label
with open(CATEGORIES, 'r') as fl:
    CLASSES = np.array(fl.read().splitlines())


def _main_():

    with open("test_images.txt", "r") as textfile:
        test_imgs = textfile.read().splitlines()
        print(test_imgs)

    with tf.Session() as sess:
        # Build Model
        darknet = darknet19(input_size=(IMG_INPUT, IMG_INPUT, 3))
        yolov2 = YOLOv2(feature_extractor=darknet, num_anchors=N_ANCHORS, num_classes=N_CLASSES,
                        fine_grain_layer=['leaky_re_lu_13'])

        yolov2.model.summary()
        yolov2.model.load_weights(WEIGHTS_PATH)

        for img_path in test_imgs:
            orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            orig_size = orig_img.shape
            img = cv2.resize(orig_img, (IMG_INPUT, IMG_INPUT))
            boxes_prediction, scores_prediction, classes_prediction = yolov2.predict(img, iou_threshold=0.5, score_threshold=0.6, mode=MODE)
            bboxes = []
            # Create a list of  bounding boxes in original image size
            for box, score, cls in zip(boxes_prediction, scores_prediction, classes_prediction):
                obj = CLASSES[cls]
                if MODE == 0:
                    obj = "Object"
                box = box * np.array(2 * [orig_size[0]/float(IMG_INPUT), orig_size[1]/float(IMG_INPUT)])
                y1, x1, y2, x2 = box
                print("Found a '{}' at {} with {:-2f}%".format(obj, (x1, y1, x2, y2), 100*score))
                bboxes.append(Box(x1, y1, x2, y2, obj, score))

            result = draw(orig_img, bboxes)
            result.save("./evaluation/"+img_path.split('/')[-1])


if __name__ == "__main__":
    _main_()