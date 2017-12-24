"""
Performance Evaluation on a image directory or a CSV file (containing path to images)

Inputs:
   A image directory with file extension as *.png:
   OR a CSV containing paths to images

Output:
   a CSV file for measuring mAP with ground truth
"""

import csv
import fnmatch
import os

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf

from config import *
from models.post_process import post_process
from models.net_builder import YOLOv2MetaArch
from models.FeatureExtractor import FeatureExtractor

from yolov2.utils import DrawingBox
from yolov2.utils import preprocess_img
from yolov2.utils import draw_bboxes

from argparse import ArgumentParser
parser = ArgumentParser(description="Over-fit one sample to validate YOLOv2 Loss Function")

parser.add_argument('-f', '--csv-file',
                    help="Path to CSV file", type=str, default=None)

parser.add_argument('-p', '--img-path',
                    help="Path to image directory", type=str, default=None)

parser.add_argument('-w', '--weights',
                    help="Path to pre-trained weight files", type=str, default=None)

parser.add_argument('-i', '--iou',
                    help="IoU value for Non-max suppression", type=float, default=0.5)

parser.add_argument('-t', '--threshold',
                    help="Threshold value to display box", type=float, default=0.5)

parser.add_argument('-o', '--output-path',
                    help="Save image to output directory", type=str, default=None)

parser.add_argument('-m', '--mode',
                    help="(Hierachical Tree Only) detection mode: 0 (Traffic Sign) - 1 (Super class) - 2 (Specific Sign)",
                    type=int, default=1)


def _main_():
    args      = parser.parse_args()
    CSV_FILE  = args.csv_file
    IMG_PATH  = args.img_path
    WEIGHTS   = args.weights
    IOU       = args.iou
    THRESHOLD = args.threshold
    OUTPUT    = args.output_path
    MODE      = args.mode

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
        darknet = FeatureExtractor(is_training=True, img_size=None, model=FEATURE_EXTRACTOR)
        yolo = YOLOv2MetaArch(num_classes=N_CLASSES,
                              anchors=np.array(ANCHORS) * (IMG_INPUT_SIZE / 608),
                              is_training=False,
                              feature_extractor=darknet,
                              detector=FEATURE_EXTRACTOR)

        yolov2 = yolo.model
        yolov2.summary()
        yolov2.load_weights(WEIGHTS)
        img_shape = K.placeholder(shape=(2,))

        # Start prediction
        boxes, classes, scores = post_process(yolov2,
                                              img_shape,
                                              n_classes=N_CLASSES,
                                              anchors=ANCHORS,
                                              iou_threshold=IOU,
                                              score_threshold=THRESHOLD,
                                              mode=MODE)

        # Save detections into detections.csv file to compare with ground truth later
        with open("detections.csv", "wb") as csv_file:
            fieldnames = ['Filename', 'x1', 'y1', 'x2', 'y2', 'annotation tag', 'probabilities']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            for instance in testing_instances:
                img_path = instance
                if not os.path.isfile(img_path):
                    continue
                orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                height, width, _ = orig_img.shape

                img = preprocess_img(cv2.resize(orig_img, (IMG_INPUT_SIZE, IMG_INPUT_SIZE)))
                img = np.expand_dims(img, 0)

                pred_bboxes, pred_classes, pred_scores = sess.run([boxes, classes, scores],
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
                    del result


def get_img_path(CSV_FILE, IMG_PATH):
    """
    Create Testing Instances to feed into network for predictions.

    Input could be a CSV file or a path to a directory

    :param CSV_FILE:
    :param IMG_PATH:
    :return:
    """
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
