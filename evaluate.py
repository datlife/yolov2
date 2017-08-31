import csv
import fnmatch
import os
import time
from argparse import ArgumentParser

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf

from cfg import *
from models.post_process import predict
from models.yolov2 import YOLOv2
from models.yolov2_mobile import MobileYOLOv2
from utils.draw_boxes import DrawingBox
from utils.preprocess_img import preprocess_img
from utils.visualize import draw_bboxes

parser = ArgumentParser(description="Over-fit one sample to validate YOLOv2 Loss Function")
parser.add_argument('-f', '--csv-file', help="Path to CSV file", type=str, default=None)
parser.add_argument('-p', '--img-path', help="Path to image directory", type=str, default=None)

parser.add_argument('-w', '--weights', help="Path to pre-trained weight files", type=str, default=None)
parser.add_argument('-i', '--iou', help="IoU value for Non-max suppression", type=float, default=0.5)
parser.add_argument('-t', '--threshold', help="Threshold value to display box", type=float, default=0.5)
parser.add_argument('-o', '--output-path', help="Save image to output directory", type=str, default=None)
parser.add_argument('-m', '--mode',
                    help="(Hierachical Tree Only) detection mode: 0 (Traffic Sign) - 1 (Super class) - 2 (Specific Sign)",
                    type=int, default=1)

ANCHORS    = np.asarray(ANCHORS).astype(np.float32)


def _main_():
    args = parser.parse_args()
    CSV_FILE = args.csv_file
    IMG_PATH = args.img_path
    WEIGHTS = args.weights
    IOU = args.iou
    THRESHOLD = args.threshold
    OUTPUT = args.output_path
    MODE = args.mode

    # Load class names
    with open(CATEGORIES, mode='r') as txt_file:
        class_names = [c.strip() for c in txt_file.readlines()]

    # Load img path
    testing_instances = get_img_path(CSV_FILE, IMG_PATH)

    # Save image to evaluation dir
    if OUTPUT is not None:
        if not os.path.exists(OUTPUT):
            os.makedirs(OUTPUT)

    with tf.Session() as sess:
        # yolov2 = YOLOv2(img_size=(IMG_INPUT, IMG_INPUT, 3), num_classes=N_CLASSES, num_anchors=len(ANCHORS))
        yolov2 = MobileYOLOv2(img_size=(IMG_INPUT, IMG_INPUT, 3), num_classes=N_CLASSES, num_anchors=N_ANCHORS)

        yolov2.load_weights(WEIGHTS)
        img_shape = K.placeholder(shape=(2,))

        # Start prediction
        boxes, classes, scores, timing = predict(yolov2, img_shape, n_classes=N_CLASSES, anchors=ANCHORS,
                                                 iou_threshold=IOU,
                                                 score_threshold=THRESHOLD,
                                                 mode=MODE)

        with open("detections.csv", "wb") as csv_file:
            fieldnames = ['Filename', 'x1', 'y1', 'x2', 'y2', 'annotation tag', 'probabilities']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            for instance in testing_instances:
                img_path = instance
                if not os.path.isfile(img_path):
                    continue
                orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                height, width, _ = orig_img.shape
                img = preprocess_img(cv2.resize(orig_img, (IMG_INPUT, IMG_INPUT)))
                img = np.expand_dims(img, 0)

                pred_bboxes, pred_classes, pred_scores = sess.run([boxes, classes, scores, timing],
                                                                  feed_dict={
                                                                      yolov2.input: img,
                                                                      img_shape: [height, width],
                                                                      K.learning_phase(): 0
                                                                  })
                bboxes = []
                for box, cls, score in zip(pred_bboxes, pred_classes, pred_scores):
                    y1, x1, y2, x2 = box
                    bboxes.append(DrawingBox(x1, y1, x2, y2, class_names[cls], score))
                    # print("Found {} with {}% on image {}".format(class_names[cls], score, img_path.split('/')[-1]))
                    writer.writerow({'Filename': img_path,
                                     'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                     'annotation tag': class_names[cls],
                                     'probabilities': score})

                # Save image to evaluation dir
                if OUTPUT is not None:
                    result = draw_bboxes(orig_img, bboxes)
                    result.save(os.path.join(OUTPUT, img_path.split('/')[-1]))
                del orig_img, img, bboxes


def get_img_path(CSV_FILE, IMG_PATH):
    testing_instances = []
    if CSV_FILE is not None:
        with open(CSV_FILE, 'rb') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                # print(row)
                testing_instances.append(row)

            # Extract image path only
            testing_instances = [i[0] for i in testing_instances[1:]]
    else:
        if IMG_PATH is None:
            raise IOError("Image path is invalid. Either enter CSV file or image path")
        IMG_PATH = os.path.abspath(IMG_PATH)

        for root, dirnames, filenames in os.walk(IMG_PATH):
            for filename in fnmatch.filter(filenames, '*.png'):
                testing_instances.append(os.path.join(root, filename))

    return testing_instances


if __name__ == "__main__":
    _main_()
    print("Done!")
