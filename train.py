"""
Training object detector for YOLOv2.

Assumption:
----------
   * A training text file has been generated.
   * A feature extractor (DarkNet19) has been pre-trained. Otherwise, it will take a long long time or might not converge

Return
-------
   A weight file `yolov2.weights` for evaluation

"""
import os
import random
from argparse import ArgumentParser

import keras
from keras.callbacks import TensorBoard
from keras.regularizers import l2

from cfg import *
from models.yolov2 import YOLOv2
from models.yolov2_loss import custom_loss
from utils.data_generator import flow_from_list
from utils.parser import parse_inputs

parser = ArgumentParser(description="Train YOLOv2")
parser.add_argument('-p', '--path', help="Path to training data set (e.g. /dataset/lisa/) ", type=str,
                    default='training.txt')
parser.add_argument('-v', '--val-path', help="Path to testing data set ", type=str, default='testing.txt')

parser.add_argument('-w', '--weights', help="Path to pre-trained weight files", type=str, default=None)
parser.add_argument('-e', '--epochs',  help='Number of epochs for training', type=int, default=10)
parser.add_argument('-i', '--start_epoch', help='Initial epoch (helpful for restart training)', type=int, default=0)
parser.add_argument('-b', '--batch',   help='Number of batch size', type=int, default=1)
parser.add_argument('-s', '--backup',  help='Path to to backup model directory', type=str, default='./backup/')
parser.add_argument('-lr','--learning_rate', type=float, default=0.000001)

args = parser.parse_args()
training_path = args.path
val_path = args.val_path
WEIGHTS_FILE    = args.weights
BATCH_SIZE      = args.batch
EPOCHS          = args.epochs
LEARNING_RATE   = args.learning_rate  # this model has been pre-trained, LOWER LR is needed
INITIAL_EPOCH = args.start_epoch
BACK_UP_PATH    = args.backup


def _main_():
    if not os.path.exists(BACK_UP_PATH):
        os.makedirs(BACK_UP_PATH)
        print("A backup directory has been created")

    # Build Model
    yolov2 = YOLOv2(img_size=(IMG_INPUT, IMG_INPUT, 3), num_classes=N_CLASSES, num_anchors=N_ANCHORS,
                    kernel_regularizer=l2(5e-8))

    # Read training input
    data = parse_inputs(training_path)
    validation_dict = parse_inputs(val_path)

    # Shuffle and load training data into a dictionary [dict[image_path] = list of objects in that image]
    shuffled_keys = random.sample(data.keys(), len(data.keys()))
    training_dict = dict([(key, data[key]) for key in shuffled_keys])

    # Construct Data Generator
    val_data_gen = flow_from_list(validation_dict, batch_size=BATCH_SIZE, augmentation=False)

    # for Debugging during training
    tf_board, backup_model = setup_debugger(yolov2)

    # Start training here
    print("Starting training process\n")
    print(
    "Hyper-parameters: LR {} | Batch {} | Optimizers {} | L2 {}".format(LEARNING_RATE, BATCH_SIZE, "Adam", "5e-8"))

    for layer in yolov2.layers[:-1]:
        layer.trainable = False
    yolov2.summary()
    model = yolov2
    # Load pre-trained file if one is available
    if WEIGHTS_FILE:
        model.load_weights(WEIGHTS_FILE)

    train_data_gen = flow_from_list(training_dict, batch_size=16, augmentation=True)

    print("Stage 1 Training...Frozen all layers except last one")
    model.compile(optimizer=keras.optimizers.adam(lr=0.000001), loss=custom_loss)
    model.fit_generator(generator=train_data_gen,
                        steps_per_epoch=int(len(training_dict) / 16),
                        validation_data=val_data_gen,
                        validation_steps=int(len(validation_dict) / BATCH_SIZE),
                        callbacks=[tf_board, backup_model],
                        epochs=8, initial_epoch=0, workers=3, verbose=1)
    model.save_weights('stage1.weights')

    for layer in yolov2.layers[:-19]:
        layer.trainable = True
    model = yolov2
    model.load_weights('stage1.weights')
    model.compile(keras.optimizers.Adam(lr=LEARNING_RATE), loss=custom_loss)
    train_data_gen = flow_from_list(training_dict, batch_size=BATCH_SIZE, augmentation=True)

    print("Stage 2 Training...Full training")
    yolov2.fit_generator(generator=train_data_gen,
                         steps_per_epoch=int(len(training_dict) / BATCH_SIZE),
                         validation_data=val_data_gen,
                         validation_steps=int(len(validation_dict) / BATCH_SIZE),
                         epochs=EPOCHS, initial_epoch=INITIAL_EPOCH,
                         callbacks=[tf_board, backup_model],
                         workers=3, verbose=1)

    yolov2.save_weights('yolov2.weights')


def setup_debugger(yolov2):
    """
    Debugger for monitoring model during training
    :param yolov2:
    :return:
    """
    tf_board     = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    backup       = keras.callbacks.ModelCheckpoint(BACK_UP_PATH+"best_model-{epoch:02d}-{val_loss:.2f}.weights",
                                                   monitor='val_loss', save_weights_only=True, save_best_only=True)
    return tf_board, backup

if __name__ == "__main__":
    _main_()