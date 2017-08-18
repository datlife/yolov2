import cv2
import numpy as np
import tensorflow as tf
from models.yolov2 import YOLOv2
from models.predict import predict
from utils.visualize import draw_bboxes
from utils.draw_boxes import DrawingBox

from cfg import *
from argparse import ArgumentParser

parser = ArgumentParser(description="Over-fit one sample to validate YOLOv2 Loss Function")
parser.add_argument('-f', '--csv-file', help="Path to CSV file", type=str, default='./test_images.csv')
parser.add_argument('-w', '--weights', help="Path to pre-trained weight files", type=str, default=None)
parser.add_argument('-i', '--iou', help="IoU value for Non-max suppression", type=float, default=0.5)
parser.add_argument('-t', '--threshold', help="Threshold value to display box", type=float, default=0.7)

ANCHORS    = np.asarray(ANCHORS).astype(np.float32)


def pre_process(img):
    img = img / 255.
    return img


def _main_():
    args = parser.parse_args()
    TEST_DATA = args.csv_file
    WEIGHTS = args.weights
    IOU = args.iou
    THRESHOLD = args.threshold

    # Load class names
    with open(CATEGORIES, mode='r') as txt_file:
        class_names = [c.strip() for c in txt_file.readlines()]

    # Load img path
    testing_instances = []
    import csv
    with open(TEST_DATA, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            print(row)
            testing_instances.append(row)

    with tf.Session() as sess:
        yolov2 = YOLOv2(img_size=(IMG_INPUT, IMG_INPUT, 3), num_classes=N_CLASSES, num_anchors=len(ANCHORS))
        yolov2.load_weights(WEIGHTS)

        # Load input
        for instance in testing_instances[1:]:
            img_path = instance[0]
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
            result = draw_bboxes(orig_img, bboxes)
            result.save('./evaluation/' + img_path.split('/')[-1])


if __name__ == "__main__":
    _main_()
    print("Done!")
