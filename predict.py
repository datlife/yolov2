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
    test_imgs  = shuffle(test_imgs)
    import fnmatch
    import os

    test_imgs = []
    for root, dirnames, filenames in os.walk('./testiamges'):
        for filename in fnmatch.filter(filenames, '*.png'):
            test_imgs.append(os.path.join(root, filename))
    test_imgs = ['/media/sharedHDD/LISA/vid6/frameAnnotations-MVI_0071.MOV_annotations/stop_1323896899.avi_image20.png',
'/media/sharedHDD/lisa_extenstion/training/2014-07-09_17-38/1/frameAnnotations-1.avi_annotations/stop_1405034298.avi_image5.png',
'/media/sharedHDD/LISA/aiua120214-0/frameAnnotations-DataLog02142012_external_camera.avi_annotations/keepRight_1330547284.avi_image0.png',
'/media/sharedHDD/LISA/vid0/frameAnnotations-vid_cmp2.avi_annotations/stop_1323803184.avi_image8.png',
'/media/sharedHDD/LISA/vid6/frameAnnotations-MVI_0071.MOV_annotations/stop_1323896946.avi_image5.png',
'/media/sharedHDD/LISA/aiua120214-1/frameAnnotations-DataLog02142012_001_external_camera.avi_annotations/addedLane_1331865841.avi_image2.png',
'/media/sharedHDD/lisa_extenstion/training/2014-05-01_17-03/2/frameAnnotations-2.avi_annotations/pedestrianCrossing_1404948415.avi_image2.png',
'/media/sharedHDD/LISA/aiua120306-1/frameAnnotations-DataLog02142012_003_external_camera.avi_annotations/turnRight_1333397687.avi_image10.png',
'/media/sharedHDD/lisa_extenstion/training/2014-05-01_16-29/2/frameAnnotations-2.avi_annotations/speedLimit35_1404942092.avi_image5.png',
'/media/sharedHDD/lisa_extenstion/training/2014-07-10_12-25/1/frameAnnotations-1.avi_annotations/keepRight_1405106337.avi_image2.png']

    with tf.Session() as sess:
        # darknet = darknet19(input_size=(IMG_INPUT, IMG_INPUT, 3))
        # yolov2 = MobileYolo(feature_extractor=darknet,
        #                     num_anchors=N_ANCHORS, num_classes=N_CLASSES,
        #                     fine_grain_layer='leaky_re_lu_13',
        #                     dropout=0.0)
        darknet = darknet19(input_size=(IMG_INPUT, IMG_INPUT, 3), freeze_layers=True)
        yolov2 = MobileYolo(feature_extractor=darknet,
                            num_anchors=N_ANCHORS, num_classes=N_CLASSES,
                            fine_grain_layer='leaky_re_lu_13',
                            dropout=0.0)
        yolov2.model.summary()
        yolov2.model.load_weights(WEIGHTS_PATH)

        for img_path in test_imgs:
            orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            orig_size = orig_img.shape
            img = cv2.resize(orig_img, (IMG_INPUT, IMG_INPUT))
            boxes_prediction, scores_prediction, classes_prediction = yolov2.predict(img, iou_threshold=0.5, score_threshold=0.6)
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