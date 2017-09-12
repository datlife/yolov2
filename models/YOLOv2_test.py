import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K

from post_process import post_process
from YOLOv2 import YOLOv2
from FeatureExtractor import FeatureExtractor

IMG_INPUT = 608
ANCHORS = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)]

IMG_PATH = '../test_imgs/multiple_stop.jpg'
CATEGORIES = '../dataset/coco/categories.txt'
WEIGHTS = '../weights/yolo-coco.weights'


if __name__ == '__main__':
    with tf.Session() as sess:
        darknet = FeatureExtractor(is_training=True, img_size=None, model='yolov2')
        yolo = YOLOv2(num_classes=80,
                      anchors=np.array(ANCHORS),
                      is_training=False,
                      feature_extractor=darknet,
                      detector='yolov2')
        if WEIGHTS:
            yolo.model.load_weights(WEIGHTS)

        yolov2 = yolo.model
        yolov2.summary()

        # Load class names
        with open(CATEGORIES, mode='r') as txt_file:
            class_names = [c.strip() for c in txt_file.readlines()]
        img_shape = K.placeholder(shape=(2,))

        # Start prediction
        boxes, classes, scores = post_process(yolov2, img_shape,
                                              n_classes=80,
                                              anchors=np.array(ANCHORS),
                                              iou_threshold=0.5,
                                              score_threshold=0.5, mode=2)

        orig_img = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)
        height, width, _ = orig_img.shape
        img = yolo.feature_extractor.preprocess(cv2.resize(orig_img, (IMG_INPUT, IMG_INPUT)))
        # img = preprocess_img(cv2.resize(orig_img, (IMG_INPUT, IMG_INPUT)))
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
            print("Found {} with {}% on image {}".format(class_names[cls], score, IMG_PATH.split('/')[-1]))
