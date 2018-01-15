"""
Training object detector for YOLOv2 on new dataset.

Assumption:
----------
   * Configuration file `cfg.yaml has been setup properly, including:
   * A training and validation CSV files in correct format have been generated. (`create_dataset.py`).
Example:
--------
python train.py \
--csv_file    dataset/pascal/training_data.csv \
--config      dataset/pascal/config.yml  \

Return
-------
  a trained model

"""
import yaml
import tensorflow as tf
from argparse import ArgumentParser

# In this file, we construct the original YOLOv2 model
from yolov2.model import YOLOv2
from yolov2.core.feature_extractors import darknet19
from yolov2.core.detectors import yolov2_detector

tf.logging.set_verbosity(tf.logging.DEBUG)


def _main_():
    parser = ArgumentParser(description="Train YOLOv2")
    parser.add_argument('--csv_file', type=str, default=None,
                        help="Path to CSV training data set")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to config file")
    # PARSE CONFIG
    args = parser.parse_args()
    cfg  = yaml.load(open(args.config, 'r'))

    # Define Keras Model
    detector = YOLOv2(is_training=True,
                      feature_extractor=darknet19,
                      detector=yolov2_detector,
                      config_dict=cfg)
    # Start training
    training_cfg  = cfg['training_params']
    detector.train(training_data = args.csv_file,
                   epochs        = training_cfg['epochs'],
                   steps_per_epoch=training_cfg['steps_per_epoch'],
                   batch_size    = training_cfg['batch_size'],
                   learning_rate = training_cfg['learning_rate'])


if __name__ == "__main__":
    _main_()
