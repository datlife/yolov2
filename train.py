"""
Training object detector for YOLOv2 on new dataset.

Assumption:
----------
   * Configuration file `cfg.yaml has been setup properly, including:
         + feature extractor type
         + image size input
         + number of classes
         + path to anchors file, categories

   * A training and validation CSV files in correct format have been generated. (could be done through `create_dataset.py`).

Return
-------


Example:

python train.py \
--train-path    dataset/pascal/training_data.csv \
--config        dataset/pascal/config.yml  \

"""
import yaml
from argparse import ArgumentParser

from yolov2.model import YOLOv2
from yolov2.core.feature_extractors import darknet19
from yolov2.core.detectors.yolov2 import yolov2_detector


def _main_():
    parser = ArgumentParser(description="Train YOLOv2")
    parser.add_argument('--csv_file', type=str, default=None,
                        help="Path to CSV training data set")

    parser.add_argument('--config', type=str, default=None,
                        help="Path to config file")

    # ###############
    # PARSE CONFIG  #
    # ###############
    args = parser.parse_args()
    cfg  = yaml.load(open(args.config, 'r'))
    training_data = args.csv_file

    # ###################
    # Define Keras Model
    # ###################
    yolov2 = YOLOv2(is_training=True , feature_extractor=darknet19, detector=yolov2_detector, config_dict=cfg)

    # ###################
    # Start training
    # ###################
    # Hyper-parameters
    training_cfg  = cfg['training_params']

    yolov2.train(training_data = training_data,
                 epochs        = training_cfg['epochs'],
                 batch_size    = training_cfg['batch_size'],
                 learning_rate = training_cfg['learning_rate'])


if __name__ == "__main__":
    _main_()
