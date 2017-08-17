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
import random, os
import keras

from keras.callbacks import TensorBoard
from keras.callbacks import LambdaCallback
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2

from cfg import *
from utils.parser import parse_inputs
from utils.split_dataset import split_data
from utils.data_generator import flow_from_list

from models.yolov2 import YOLOv2
from models.yolov2_loss import custom_loss

from argparse import ArgumentParser
parser = ArgumentParser(description="Train YOLOv2")
parser.add_argument('-p', '--path', help="Path to training data set (e.g. /dataset/lisa/ ", type=str,  default='training.txt')
parser.add_argument('-w', '--weights', help="Path to pre-trained weight files", type=str, default=None)
parser.add_argument('-e', '--epochs',  help='Number of epochs for training', type=int, default=10)
parser.add_argument('-b', '--batch',   help='Number of batch size', type=int, default=1)
parser.add_argument('-s', '--backup',  help='Path to to backup model directory', type=str, default='./backup/')
parser.add_argument('-lr','--learning_rate', type=float, default=0.000001)

args = parser.parse_args()
annotation_path = args.path
WEIGHTS_FILE    = args.weights
BATCH_SIZE      = args.batch
EPOCHS          = args.epochs
LEARNING_RATE   = args.learning_rate  # this model has been pre-trained, LOWER LR is needed
BACK_UP_PATH    = args.backup


def learning_rate_schedule(epochs):
    if epochs <= 60:
        return LEARNING_RATE
    if 60 < epochs < 90:
        return LEARNING_RATE / 30.
    if 90 <= epochs < 160:
        return LEARNING_RATE / 300.
    if epochs >= 160:
        return LEARNING_RATE / 300.


def _main_():

    if not os.path.exists(BACK_UP_PATH):
        os.makedirs(BACK_UP_PATH)
        print("A backup directory has been created")

    # Build Model
    yolov2 = YOLOv2(img_size=(IMG_INPUT, IMG_INPUT, 3), num_classes=N_CLASSES, num_anchors=N_ANCHORS, kernel_regularizer=l2(5e-6))

    # Load pre-trained file if one is available
    if WEIGHTS_FILE:
        yolov2.load_weights(WEIGHTS_FILE, by_name=True)

    yolov2.summary()

    # Read training input
    # training, testing = split_data(annotation_path, ratio=0.1)
    data = parse_inputs(annotation_path)
    # validation_dict = parse_inputs('./data/testing_data.csv')

    # Shuffle and load training data into a dictionary [dict[image_path] = list of objects in that image]
    shuffled_keys = random.sample(data.keys(), len(data.keys()))
    training_dict = dict([(key, data[key]) for key in shuffled_keys])
    training_dict = dict(training_dict.items()[0:20])
    with open("test_images.csv", "wb") as csv_file:
        fieldnames = ['Filename', 'annotation tag', 'x1','y1', 'x2', 'y2']
        import csv
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for fname in training_dict:
            gts = training_dict[fname]
            for gt in gts:
                box, label = gt
                xc, yc, w, h = box.to_array()
                x1 = xc - 0.5*w
                y1 = yc - 0.5*h
                x2 = xc + 0.5*w
                y2 = yc + 0.5*h
                box = "%s, %s, %s, %s"%(x1, y1, x2, y2)
                writer.writerow({'Filename': fname, 'annotation tag': label,
                                 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
                print("{}, {}, {}\n".format(fname, box, label))
    # Construct Data Generator
    train_data_gen = flow_from_list(training_dict, batch_size=BATCH_SIZE, augmentation=False)
    # val_data_gen   = flow_from_list(validation_dict, batch_size=BATCH_SIZE)

    # for Debugging during training
    tf_board, lr_scheduler, backup_model = setup_debugger(yolov2)

    sgd  = keras.optimizers.SGD(lr=LEARNING_RATE, momentum=0.9, decay=0.0005)
    yolov2.compile(optimizer=sgd, loss=custom_loss)

    # Start training here
    print("Starting training process\n")
    print("Hyper-parameters: LR {} | Batch {} | Optimizers {} | L2 {}".format(LEARNING_RATE, BATCH_SIZE, "SGD", "5e-6"))
    yolov2.fit_generator(generator=train_data_gen,
                         steps_per_epoch=20*int(len(training_dict)/BATCH_SIZE),
                         # validation_data=val_data_gen,
                         # validation_steps=int(len(validation_dict)/BATCH_SIZE),
                         epochs=EPOCHS, initial_epoch=0,
                         callbacks=[tf_board, lr_scheduler],
                         workers=3, verbose=1)
    print(training_dict)

    yolov2.save_weights('yolov2.weights')


def setup_debugger(yolov2):
    """
    Debugger for monitoring model during training
    :param yolov2:
    :return:
    """
    lr_scheduler = LearningRateScheduler(learning_rate_schedule)
    tf_board     = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    backup       = keras.callbacks.ModelCheckpoint(BACK_UP_PATH+"best_model-{epoch:02d}-{val_loss:.2f}.weights",
                                                   monitor='val_loss', save_weights_only=True, save_best_only=True)
    return tf_board, lr_scheduler, backup

if __name__ == "__main__":
    _main_()
    # with open("test_images.csv", "wb") as csv_file:
    #     fieldnames=['Filename','annotation tag', 'x1','y1', 'x2', 'y2']
    #     import csv
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     for fname in training_dict:
    #         gts = training_dict[fname]
    #         for gt in gts:
    #             box, label = gt
    #             xc, yc, w, h = box.to_array()
    #             x1 = xc - 0.5*w
    #             y1 = yc - 0.5*h
    #             x2 = xc + 0.5*w
    #             y2 = yc + 0.5*h
    #             box = "%s, %s, %s, %s"%(x1, y1, x2, y2)
    #             writer.writerow({'Filename': fname, 'annotation tag': label,
    #                              'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    #             print("{}, {}, {}\n".format(fname, box, label))
    # test