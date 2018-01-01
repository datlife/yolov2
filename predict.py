"""
Predict one image (for testing)
"""
import os
import cv2
import yaml
import argparse
import numpy as np

from yolov2.zoo import yolov2_darknet19
from yolov2.utils.painter import draw_boxes
from yolov2.utils.parser import parse_label_map


def _main_():
    parser = argparse.ArgumentParser(description="Detect objects in an image")

    parser.add_argument('--path', type=str, default='./test_imgs/person.jpg',
                        help="Path to image file")

    parser.add_argument('--weights', type=str, default='yolo-coco.weights',
                        help="Path to pre-trained weight file")

    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output Directory")

    parser.add_argument('--iou', type=float, default=0.5,
                        help="Intersection over Union (IoU) value")

    parser.add_argument('--threshold', type=float, default=0.5,
                        help="Score Threshold value (minimum accuracy)")

    # ############
    # Parse Config
    # ############
    args = parser.parse_args()
    with open('config.yml', 'r') as stream:
        config = yaml.load(stream)

    anchors    = np.array(config['anchors'])
    label_dict = parse_label_map(config['label_map'])

    # ###################
    # Define Keras Model
    # ###################
    model_cfg  = config['model']
    model = yolov2_darknet19(is_training      = False,
                             img_size         = model_cfg['image_size'],
                             anchors          = anchors,
                             num_classes      = model_cfg['num_classes'],
                             iou              = args.iou,
                             scores_threshold = args.threshold,
                             max_boxes        = 100)

    model.load_weights(args.weights)
    model.summary()

    # #####################
    # Make one prediction #
    # #####################
    image = cv2.imread(args.path)

    bboxes, scores, classes = model.predict_on_batch(np.expand_dims(image, axis=0))

    # Convert idx to label
    classes = [label_dict[idx] for idx in classes]

    # #################
    # Display Result  #
    # #################
    h, w, _ = image.shape
    if args.output_dir is not None:
        # Scale relative coordinates into actual coordinates
        bboxes  = [box * np.array([h, w, h, w]) for box in bboxes]
        result = draw_boxes(image, bboxes, classes, scores)
        cv2.imwrite(os.path.join(args.output_dir, args.path.split('/')[-1].split('.')[0] + '_result.jpg'), result)


if __name__ == "__main__":
    _main_()
