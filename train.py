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
import numpy as np
from argparse import ArgumentParser

import keras
import keras.backend as K
from keras.layers import Input
from keras.models import Model

from config import *
from models.yolov2_darknet import yolov2_darknet
from yolov2.utils import parse_inputs

parser = ArgumentParser(description="Train YOLOv2")
parser.add_argument('--train-path',
                    help="Path to CSV training data set", type=str, default=None)

parser.add_argument('--weights',
                    help="Path to pre-trained weight files", type=str, default=None)

parser.add_argument('--epochs',
                    help='Number of epochs for training', type=int, default=10)

parser.add_argument('--batch_size' , type=int, default=1,
                    help='Number of batch size')

parser.add_argument('--learning_rate', type=float, default=0.00001)

parser.add_argument('--initial_epoch',
                    help='Initial epoch (helpful for restart training)', type=int, default=0)

parser.add_argument('--backup_dir' , type=str, default='./backup/',
                    help='Path to to backup model directory')


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
    weight_file   = args.weights
    batch_size    = args.batch
    epochs        = args.epochs
    learning_rate = args.learning_rate  # this model has been pre-trained, LOWER LR is needed
    initial_epoch = args.initial_epoch
    backup_dir  = args.backup_dir

    # Config Anchors and encoding dict
    anchors, label_dict = config_prediction()

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print("A backup directory has been created")

    # ###############
    # PREPARE DATA  #
    # ###############
    # Read training input (a CSV file)
    data   = parse_inputs(training_path, label_dict)

    # Load X, Y
    X, Y = data.keys, data

    with K.get_session() as sess:
        inputs                 = Input(shape=(None, None, 3), name='image_input')

        # ###################
        # Define Keras Model
        # ###################
        outputs = yolov2_darknet(is_training= True,
                                 inputs     = inputs,
                                 img_size   = IMG_INPUT_SIZE,
                                 anchors    = anchors,
                                 num_classes= N_CLASSES)

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

        if weight_file:
            model.load_weights(weight_file)
            print("Weight file has been loaded in to model")

        # Set up callbacks
        tf_board = keras.callbacks.TensorBoard(log_dir       = './logs',
                                               histogram_freq= 1,
                                               write_graph   = True,
                                               write_images  = True)

        backup = keras.callbacks.ModelCheckpoint(backup_dir +
                                                 "best_%s-{epoch:02d}-{val_loss:.2f}.weights" % FEATURE_EXTRACTOR,
                                                 monitor          = 'val_loss',
                                                 save_weights_only= True,
                                                 save_best_only   = True)

        # ###################
        # COMPILE AND TRAIN #
        # ###################
        model.compile(optimizer= keras.optimizers.Adam(lr=learning_rate),
                      loss     = detection_model.loss_func)

        # @TODO: Create K-fold cross validation training procedure

        model.save_weights('yolov2.weights')
        model.fit_generator()


def config_prediction():
    # @TODO: this is ugly.

    # Config Anchors
    anchors = []
    with open(ANCHORS, 'r') as f:
        data = f.read().splitlines()
        for line in data:
            numbers = re.findall('\d+.\d+', line)
            anchors.append((float(numbers[0]), float(numbers[1])))

    # Load class names
    with open(CATEGORIES, mode='r') as txt_file:
        class_names = [c.strip() for c in txt_file.readlines()]

    label_dict = {v: k for v, k in enumerate(class_names)}
    return np.array(anchors), label_dict


if __name__ == "__main__":
    _main_()
