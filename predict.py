import os
import cv2
import argparse
import numpy as np
import config as cfg

from yolov2.models import yolov2_darknet
from yolov2.utils import draw, parse_config


def _main_():
    parser = argparse.ArgumentParser(description="Detect object in an image")

    parser.add_argument('--path', type=str, default='./test/test_imgs/person.jpg',
                        help="Path to image file")

    parser.add_argument('--weights', type=str, default='yolo-coco.weights',
                        help="Path to pre-trained weight file")

    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output Directory")

    parser.add_argument('--iou', type=float, default=0.5,
                        help="Intersection over Union (IoU) value")

    parser.add_argument('--threshold', type=float, default=0.6,
                        help="Score Threshold value (minimum accuracy)")

    # ############
    # Parse Config
    # ############
    args = parser.parse_args()
    anchors, label_dict = parse_config(cfg)

    # ###################
    # Define Keras Model
    # ###################
    model = yolov2_darknet(is_training      = False,
                           img_size         = cfg.IMG_INPUT_SIZE,
                           anchors          = anchors,
                           num_classes      = cfg.N_CLASSES,
                           iou              = args.iou,
                           scores_threshold = args.threshold)

    model.load_weights(args.weights)
    model.summary()

    # #####################
    # Make one prediction #
    # #####################
    image = cv2.imread(args.path)

    pred_bboxes, pred_classes, pred_scores = model.predict_on_batch(np.expand_dims(image, axis=0))

    # Convert idx to label
    pred_classes = [label_dict[idx] for idx in pred_classes]

    # #################
    # Display Result  #
    # #################
    h, w, _ = image.shape
    if args.output_dir is not None:
        # Scale relative coordinates into actual coordinates
        pred_bboxes  = [box * np.array([h, w, h, w]) for box in pred_bboxes]
        print("pred_bboxes = {}".format(pred_bboxes))
        print("pred_classes = {}".format(pred_classes))
        print("pred_scores = {}".format(pred_scores))

        result = draw(image, pred_bboxes, pred_classes, pred_scores)
        cv2.imwrite(os.path.join(args.output_dir, args.path.split('/')[-1].split('.')[0] + '_result.jpg'), result)


if __name__ == "__main__":
    _main_()
