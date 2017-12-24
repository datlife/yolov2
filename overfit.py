"""
Overfit one image with 1000 epochs to test the loss function properly
"""
import random
import keras
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser

from cfg import *
from models.net_builder import YOLOv2MetaArch
from models.FeatureExtractor import FeatureExtractor

from utils.parser import parse_inputs
from models.loss import custom_loss
from utils.data_generator import flow_from_list

parser = ArgumentParser(description="Over-fit one sample to validate YOLOv2 Loss Function")

parser.add_argument('-p', '--path', help="Path to training text file ",
                    type=str, default=None)

parser.add_argument('-w', '--weights', help="Path to pre-trained weight files",
                    type=str, default=None)

parser.add_argument('-lr', '--learning_rate',
                    type=float, default=0.001)

parser.add_argument('-e', '--epochs', help='Number of epochs for training',
                    type=int, default=1000)

parser.add_argument('-b', '--batch', help='Number of batch size',
                    type=int, default=1)

args = parser.parse_args()
annotation_path = args.path
WEIGHTS_FILE    = args.weights
BATCH_SIZE      = args.batch
EPOCHS          = args.epochs
LEARNING_RATE   = args.learning_rate  # this model has been pre-trained, LOWER LR is needed


def _main_():

    # ###################
    # PREPARE DATA INPUT
    # ###################
    size = 1

    data = parse_inputs(annotation_path)
    shuffled_keys = random.sample(data.keys(), len(data.keys()))
    training_dict = dict([(key, data[key]) for key in shuffled_keys])

    # Create one instance for over-fitting model
    i = np.random.randint(0, len(training_dict))
    training_dict = dict(training_dict.items()[i:i + size])

    # #################
    # CONSTRUCT MODEL
    # #################
    with tf.variable_scope('yolov2', regularizer=None):
        darknet = FeatureExtractor(is_training=True, img_size=None, model=FEATURE_EXTRACTOR)
        yolo = YOLOv2MetaArch(num_classes=N_CLASSES,
                              anchors=np.array(ANCHORS),
                              is_training=False,
                              feature_extractor=darknet,
                              detector=FEATURE_EXTRACTOR)

        for l in yolo.feature_extractor.model.layers:
            l.trainable = False

        model = yolo.model
        if WEIGHTS_FILE:
            model.load_weights(WEIGHTS_FILE, by_name=True)
        model.summary()

        # #################
        # COMPILE AND RUN
        # #################
        model.compile(keras.optimizers.Adam(lr=0.001), loss=custom_loss)
        train_data_gen = flow_from_list(training_dict, batch_size=size)
        model.fit_generator(generator=train_data_gen, steps_per_epoch=len(training_dict) / size,
                            epochs=EPOCHS, workers=3,
                            verbose=1)

        print(training_dict)
        model.save_weights('overfit.weights')

if __name__ == "__main__":
    _main_()