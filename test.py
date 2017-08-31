import cv2
import numpy as np
import tensorflow as tf

from cfg import *
from models.post_process import predict
from models.yolov2 import YOLOv2
from utils.draw_boxes import DrawingBox
from utils.visualize import draw_bboxes

IMG_PATH   = '/home/dat/Documents/lisa_extension/training/2014-07-11_13-17/2/frameAnnotations-2.avi_annotations/keepRight_1405362361.avi_image1.png'
WEIGHTS    = './weights/yolo-coco.weights'
ANCHORS    = np.asarray(ANCHORS).astype(np.float32)


def pre_process(img):
    img = img / 255.
    return img


def _main_():

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
                                         iou_threshold=0.5, score_threshold=0.6)

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
