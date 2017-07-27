import cv2
import os
import tensorflow as tf
import keras.backend as K
from model.densenet import DenseNet
from model.darknet19 import darknet19
from model.MobileYolo import MobileYolo
from sklearn.utils import shuffle
from utils.draw_boxes import Box, draw
from utils.parse_txt_to_inputs import parse_txt_to_inputs
from cfg import *
K.clear_session()  # to avoid duplicating model

WEIGHTS_PATH = './yolov2.weights'

# Map int to label
with open(CATEGORIES, 'r') as fl:
    CLASSES = np.array(fl.read().splitlines())


def _main_():
    test_imgs, _ = parse_txt_to_inputs('training_extension.txt')
    test_imgs  = shuffle(test_imgs)[0:30]

    yolov2  = MobileYolo(feature_extractor=darknet19(), num_anchors=N_ANCHORS, num_classes=N_CLASSES, fine_grain_layer='leaky_re_lu_13')
    yolov2.model.load_weights(WEIGHTS_PATH)

    for img_path in test_imgs:
        orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        orig_size = orig_img.shape
        img = cv2.resize(orig_img, (IMG_INPUT, IMG_INPUT))
        boxes_prediction, scores_prediction, classes_prediction = yolov2.predict(img, iou_threshold=0.5, score_threshold=0.1)
        bboxes = []

        # Create a list of  bounding boxes in original image size
        for box, score, cls in zip(boxes_prediction, scores_prediction, classes_prediction):
            box = box * np.array(2 * [orig_size[0]/float(IMG_INPUT), orig_size[1]/float(IMG_INPUT)])
            y1, x1, y2, x2 = box
            print("Found a '{}' at {} with {:-2f}%".format(CLASSES[cls], (x1, y1, x2, y2), 100*score))
            bboxes.append(Box(x1, y1, x2, y2, CLASSES[cls], score))

        result = draw(orig_img, bboxes)
        result.save("./evalutation/"+img_path.split('/')[-1])

    # plt.figure(figsize=(15, 15))
    # plt.imshow(result)

if __name__ == "__main__":
    _main_()