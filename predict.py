import cv2
import os
import fnmatch

import tensorflow as tf
import keras.backend as K
from sklearn.utils import shuffle

from model.darknet19 import darknet19
from model.YOLOv2 import YOLOv2
from model.MobileYolo import MobileYolo

from cfg import *
from utils.draw_boxes import Box, draw
K.clear_session()  # to avoid duplicating model

WEIGHTS_PATH = './yolov2.weights'

# Map int to label
with open(CATEGORIES, 'r') as fl:
    CLASSES = np.array(fl.read().splitlines())


def _main_():
    # test_imgs, _ = parse_txt_to_inputs('training_extension.txt')
    # test_imgs  = shuffle(test_imgs)

    test_imgs = []
    # for root, dirnames, filenames in os.walk('./testiamges'):
    #     for filename in fnmatch.filter(filenames, '*.png'):
    #         test_imgs.append(os.path.join(root, filename))

    with open("test_images.txt", "r") as textfile:
        test_imgs = textfile.read().splitlines()
        print(test_imgs)

    with tf.Session() as sess:
        # Build Model
        darknet = darknet19(input_size=(IMG_INPUT, IMG_INPUT, 3))
        yolov2 = MobileYolo(feature_extractor=darknet, num_anchors=N_ANCHORS, num_classes=N_CLASSES,
                            fine_grain_layer=['leaky_re_lu_13', 'leaky_re_lu_8'], dropout=None)
        #
        # darknet = darknet19(input_size=(IMG_INPUT, IMG_INPUT, 3))
        # yolov2 = YOLOv2(feature_extractor=darknet, num_anchors=N_ANCHORS, num_classes=N_CLASSES,
        #                 fine_grain_layer=['leaky_re_lu_13'])

        yolov2.model.summary()
        yolov2.model.load_weights(WEIGHTS_PATH)

        for img_path in test_imgs:
            orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            orig_size = orig_img.shape
            img = cv2.resize(orig_img, (IMG_INPUT, IMG_INPUT))
            boxes_prediction, scores_prediction, classes_prediction = yolov2.predict(img, iou_threshold=0.5, score_threshold=0.3, mode=0)
            bboxes = []
            # Create a list of  bounding boxes in original image size
            for box, score, cls in zip(boxes_prediction, scores_prediction, classes_prediction):
                box = box * np.array(2 * [orig_size[0]/float(IMG_INPUT), orig_size[1]/float(IMG_INPUT)])
                y1, x1, y2, x2 = box
                print("Found a '{}' at {} with {:-2f}%".format(CLASSES[cls], (x1, y1, x2, y2), 100*score))
                bboxes.append(Box(x1, y1, x2, y2, CLASSES[cls], score))

            result = draw(orig_img, bboxes)
            result.save("./evalutation/"+img_path.split('/')[-1])


if __name__ == "__main__":
    _main_()