import cv2
import os
import numpy as np
import tensorflow as tf
from models.yolov2 import YOLOv2
from models.predict import predict
from utils.visualize import draw_bboxes
from utils.draw_boxes import DrawingBox
from cfg import *

import argparse

parser = argparse.ArgumentParser("Over-fit model to validate loss function")

parser.add_argument('-p', '--path', help="Path to image file", type=str, default=None)
parser.add_argument('-w', '--weights', help="Path to pre-trained weight files", type=str, default=None)
parser.add_argument('-o', '--output-path', help="Save image to output directory", type=str, default=None)
parser.add_argument('-i', '--iou', help="IoU value for Non-max suppression", type=float, default=0.5)
parser.add_argument('-t', '--threshold', help="Threshold value to display box", type=float, default=0.7)

ANCHORS = np.asarray(ANCHORS).astype(np.float32)


def pre_process(img):
    img = img / 255.
    return img


def _main_():
    args = parser.parse_args()

    IMG_PATH = args.path
    WEIGHTS = args.weights
    OUTPUT = args.output_path
    IOU = args.iou
    THRESHOLD = args.threshold

    if not os.path.isfile(IMG_PATH):
        print("Image path is invalid.")
        exit()
    if not os.path.isfile(WEIGHTS):
        print("Weight file is invalid")
        exit()

    # Load class names
    with open(CATEGORIES, mode='r') as txt_file:
        class_names = [c.strip() for c in txt_file.readlines()]

    with tf.Session() as sess:
        yolov2 = YOLOv2(img_size=(IMG_INPUT, IMG_INPUT, 3), num_classes=N_CLASSES, num_anchors=len(ANCHORS))
        yolov2.load_weights(WEIGHTS)

        # Load input
        img_path = IMG_PATH
        orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        height, width, _ = orig_img.shape
        img = pre_process(cv2.resize(orig_img, (IMG_INPUT, IMG_INPUT)))

        # Start prediction
        boxes, classes, scores = predict(yolov2, img, n_classes=N_CLASSES, anchors=ANCHORS,
                                         iou_threshold=IOU, score_threshold=THRESHOLD)

        bboxes = []
        for box, cls, score in zip(boxes, classes, scores):
            y1, x1, y2, x2 = box * np.array(2 * [height / float(IMG_INPUT), width / float(IMG_INPUT)])
            bboxes.append(DrawingBox(x1, y1, x2, y2, class_names[cls], score))
            print("Found {} with {}%".format(class_names[cls], score))

        # Save images to disk
        if OUTPUT is not None:
            result = draw_bboxes(orig_img, bboxes)
            if not os.path.exists(OUTPUT):
                os.makedirs(OUTPUT)
                print("A evaluation directory has been created")
            result.save(os.path.join(OUTPUT, img_path.split('/')[-1]))
            print("Output has been saved to {}.".format(OUTPUT))


if __name__ == "__main__":
    _main_()
    print("Done!")
