"""
Training object detector for YOLOv2 on new dataset.

Assumption:
----------
   * Configuration file `cfg.py` has been setup properly, including:
         + feature extractor type
         + image size input
         + number of classes
         + path to anchors file, categories

   * A training and validation CSV files in correct format have been generated. (could be done through `create_dataset.py`).
   * A feature extractor has been pre-trained (darknet19, densenet or mobilenet in this case).

Return
-------
   A weight file - `yolov2.weights` for evaluation


Example:

python train.py \
--train-path    dataset/combined_lisa/training_data.csv \
--val-path      dataset/combined_lisa/testing_data.csv  \
--weights       weights/yolo-coco.weights \
--batch         8 \
--epochs        200 \
--learning_rate 0.001

"""
import os
import re
import random
import numpy as np
from argparse import ArgumentParser

import keras
from keras.callbacks import TensorBoard

from cfg import *
from models.loss import custom_loss
from models.FeatureExtractor import FeatureExtractor
from models.YOLOv2 import YOLOv2
from utils.data_generator import flow_from_list
from utils.parser import parse_inputs

parser = ArgumentParser(description="Train YOLOv2")
parser.add_argument('-p', '--train-path',
                    help="Path to CSV training data set", type=str, default=None)

parser.add_argument('-v', '--val-path',
                    help="Path to CSV validation data set ", type=str, default=None)

parser.add_argument('-w', '--weights',
                    help="Path to pre-trained weight files", type=str, default=None)

parser.add_argument('-e', '--epochs',
                    help='Number of epochs for training', type=int, default=10)

parser.add_argument('-i', '--initial_epoch',
                    help='Initial epoch (helpful for restart training)', type=int, default=0)

parser.add_argument('-b', '--batch',
                    help='Number of batch size', type=int, default=1)

parser.add_argument('-s', '--backup',
                    help='Path to to backup model directory', type=str, default='./backup/')

parser.add_argument('-lr', '--learning_rate',
                    type=float, default=0.00001)


DEFAULT_FEATURE_EXTRACTORS_WEIGHTS = {
    'yolov2': './weights/feature_extractors/darknet19_448.weights',
    'mobilenet': './weights/feature_extractors/mobilenet.h5',
    'densenet': './weights/feature_extractors/densenet201.h5',
}

def _main_():
    # ###############
    # PARSE CONFIG  #
    # ###############
    args          = parser.parse_args()
    training_path = args.train_path
    val_path      = args.val_path
    WEIGHTS_FILE  = args.weights
    BATCH_SIZE    = args.batch
    EPOCHS        = args.epochs
    LEARNING_RATE = args.learning_rate  # this model has been pre-trained, LOWER LR is needed
    INITIAL_EPOCH = args.initial_epoch
    BACK_UP_PATH  = args.backup

    # Config Anchors
    anchors = []
    with open(ANCHORS, 'r') as f:
      data = f.read().splitlines()
      for line in data:
        numbers = re.findall('\d+.\d+', line)
        anchors.append((float(numbers[0]), float(numbers[1])))
    anchors = np.array(anchors)

    if not os.path.exists(BACK_UP_PATH):
        os.makedirs(BACK_UP_PATH)
        print("A backup directory has been created")

    # ###############
    # PREPARE DATA  #
    # ###############
    # Read training input (a CSV file)
    data = parse_inputs(training_path)
    validation_dict = parse_inputs(val_path)

    # Shuffle and load training data into a dictionary [dict[image_path] = list of objects in that image]
    shuffled_keys = random.sample(data.keys(), len(data.keys()))
    training_dict = dict([(key, data[key]) for key in shuffled_keys])

    # Set up data generator
    train_data_gen = flow_from_list(training_dict, batch_size=BATCH_SIZE, augmentation=True, use_tree=False)
    val_data_gen   = flow_from_list(validation_dict, batch_size=BATCH_SIZE, augmentation=False, use_tree=False)

    # #################
    # Construct Model #
    # #################

    # set up feature extractor
    feature_extractor = FeatureExtractor(is_training=True, img_size=None, model=FEATURE_EXTRACTOR)

    if FEATURE_EXTRACTOR in DEFAULT_FEATURE_EXTRACTORS_WEIGHTS \
            and WEIGHTS_FILE is None:
        feature_extractor.model.load_weights(DEFAULT_FEATURE_EXTRACTORS_WEIGHTS[FEATURE_EXTRACTOR], by_name=True)

    for l in feature_extractor.model.layers:
        l.trainable = False

    # set up detection model
    detection_model = YOLOv2(num_classes      = N_CLASSES,
                             anchors          = anchors * float((IMG_INPUT_SIZE / 608.)),
                             is_training      = True,
                             feature_extractor= feature_extractor,
                             detector         = FEATURE_EXTRACTOR,
                             use_hierarchical_tree=False)

    model = detection_model.model
    # Load pre-trained file if one is available
    if WEIGHTS_FILE:
        model.load_weights(WEIGHTS_FILE, by_name=True)

    # Set up Tensorboard and Model Backup
    tf_board = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)
    backup = keras.callbacks.ModelCheckpoint(BACK_UP_PATH + "best_%s-{epoch:02d}-{val_loss:.2f}.weights" % FEATURE_EXTRACTOR,
                                             monitor='val_loss',
                                             save_weights_only=True,
                                             save_best_only=True)

    # ###################
    # COMPILE AND TRAIN #
    # ###################
    model.summary()
    print("Starting training process\n")
    print("Hyper-parameters: LR {} | Batch {} | Optimizers {} | L2 {}".format(LEARNING_RATE, BATCH_SIZE, "Adam", "5e-4"))
    print("Number of classes: %s" % N_CLASSES)
    print("Number of anchors: %s" % len(anchors))

    model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE), loss=detection_model.loss_func)
    model.fit_generator(generator       = train_data_gen,
                        validation_data = val_data_gen,
                        steps_per_epoch = int(len(training_dict) / BATCH_SIZE),
                        validation_steps= int(len(validation_dict) / BATCH_SIZE),
                        callbacks       = [tf_board, backup],
                        epochs          = EPOCHS,
                        initial_epoch   = INITIAL_EPOCH,
                        workers=3, verbose=1)

    model.save_weights('yolov2.weights')


if __name__ == "__main__":
    _main_()
