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
    test_imgs = ['/media/sharedHDD/LISA/aiua120306-0/frameAnnotations-DataLog02142012_002_external_camera.avi_annotations/stop_1333392807.avi_image10.png',
                 '/media/sharedHDD/LISA/vid4/frameAnnotations-vid_cmp2.avi_annotations/speedLimit_1323822585.avi_image12.png',
                 '/media/sharedHDD/LISA/aiua120306-1/frameAnnotations-DataLog02142012_003_external_camera.avi_annotations/signalAhead_1333397203.avi_image10.png',
                 '/media/sharedHDD/LISA/aiua120214-2/frameAnnotations-DataLog02142012_002_external_camera.avi_annotations/stop_1331866288.avi_image5.png',
                 '/media/sharedHDD/LISA/vid2/frameAnnotations-vid_cmp2.avi_annotations/speedLimit_1323817452.avi_image12.png',
                 '/media/sharedHDD/LISA/aiua120306-0/frameAnnotations-DataLog02142012_002_external_camera.avi_annotations/speedLimit25_1333392931.avi_image5.png',
                 '/media/sharedHDD/LISA/vid6/frameAnnotations-MVI_0071.MOV_annotations/pedestrian_1323896918.avi_image4.png',
                 '/media/sharedHDD/LISA/aiua120214-2/frameAnnotations-DataLog02142012_002_external_camera.avi_annotations/speedLimit_1331866345.avi_image20.png',
                '/media/sharedHDD/LISA/aiua120214-0/frameAnnotations-DataLog02142012_external_camera.avi_annotations/laneEnds_1330547145.avi_image19.png',
                '/media/sharedHDD/LISA/vid5/frameAnnotations-vid_cmp2.avi_annotations/speedLimit_1323824852.avi_image7.png']

    with tf.Session() as sess:
        darknet = darknet19(input_size=(IMG_INPUT, IMG_INPUT, 3))
        yolov2 = MobileYolo(feature_extractor=darknet,
                            num_anchors=N_ANCHORS, num_classes=N_CLASSES,
                            fine_grain_layer='leaky_re_lu_13',
                            dropout=0.3)
        # densenet = DenseNet(reduction=0.5, freeze_layers=True, weights_path='./weights/densenet121_weights_tf.h5')
        # yolov2 = MobileYolo(feature_extractor=densenet, num_anchors=N_ANCHORS, num_classes=N_CLASSES,
        #                     fine_grain_layer='conv4_blk')
        yolov2.model.summary()
        yolov2.model.load_weights(WEIGHTS_PATH)

        for img_path in test_imgs:
            orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            orig_size = orig_img.shape
            img = cv2.resize(orig_img, (IMG_INPUT, IMG_INPUT))
            boxes_prediction, scores_prediction, classes_prediction = yolov2.predict(img, iou_threshold=0.5, score_threshold=0.8)
            bboxes = []

            # Create a list of  bounding boxes in original image size
            for box, score, cls in zip(boxes_prediction, scores_prediction, classes_prediction):
                box = box * np.array(2 * [orig_size[0]/float(IMG_INPUT), orig_size[1]/float(IMG_INPUT)])
                y1, x1, y2, x2 = box
                print("Found a '{}' at {} with {:-2f}%".format(cls, (x1, y1, x2, y2), 100*score))
                bboxes.append(Box(x1, y1, x2, y2, cls, score))

            result = draw(orig_img, bboxes)
            result.save("./evalutation/"+img_path.split('/')[-1])

    # plt.figure(figsize=(15, 15))
    # plt.imshow(result)

if __name__ == "__main__":
    _main_()