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
import cv2
import numpy as np
import config as cfg

import keras
import tensorflow as tf
import keras.backend as K

from yolov2.zoo import yolov2_darknet19
from yolov2.core.loss import yolov2_loss

from yolov2.utils.parser import parse_config, parse_inputs
from yolov2.utils.callbacks import create_callbacks

from argparse import ArgumentParser

parser = ArgumentParser(description="Train YOLOv2")
parser.add_argument('--csv_file',
                    help="Path to CSV training data set", type=str, default=None)

parser.add_argument('--weights',
                    help="Path to pre-trained weight file", type=str, default=None)

parser.add_argument('--epochs',
                    help='Number of epochs for training', type=int, default=10)

parser.add_argument('--batch_size', type=int, default=1,
                    help='Number of batch size')

parser.add_argument('--learning_rate', type=float, default=0.00001)


parser.add_argument('--backup_dir', type=str, default='./backup/',
                    help='Path to to backup model directory')


def _main_():
    # ###############
    # PARSE CONFIG  #
    # ###############
    args          = parser.parse_args()
    training_path = args.csv_file
    weight_file   = args.weights
    batch_size    = args.batch_size
    epochs        = args.epochs
    learning_rate = args.learning_rate  # this model has been pre-trained, LOWER learning rate is needed
    backup_dir    = args.backup_dir

    # Config Anchors and encoding dict
    anchors, label_dict = parse_config(cfg)

    # ###############
    # PREPARE DATA  #
    # ###############
    # @TODO: use TFRecords ?
    # @TODO: Create K-fold cross validation training procedure
    inputs, labels = parse_inputs(training_path, label_dict)

    # ###################
    # Define Keras Model
    # ###################
    model = yolov2_darknet19(is_training= True,
                             img_size   = cfg.IMG_INPUT_SIZE,
                             anchors    = anchors,
                             num_classes= cfg.N_CLASSES)

    if weight_file:
        model.load_weights(weight_file)
        print("Weight file has been loaded in to model")

    # ###################
    # COMPILE AND TRAIN #
    # ###################
    model.compile(optimizer= keras.optimizers.Adam(lr=learning_rate),
                  loss     = yolov2_loss(anchors, cfg.N_CLASSES))

    from sklearn.model_selection import train_test_split

    for current_epoch in range(epochs):

        # Create 10-fold split
        x_train, x_val = train_test_split(inputs, test_size=0.2)
        y_train = [labels[k] for k in x_train]
        y_val   = [labels[k] for k in x_val]

        history = model.fit_generator(generator       = data_generator(x_train, y_train),
                                      steps_per_epoch = 1000,
                                      validation_data = data_generator(x_val, y_val),
                                      validation_steps= 100,
                                      verbose=1,
                                      workers=0)


def data_generator(images, labels, shuffle=True, batch_size=128):
    batch = input_func(images, labels, shuffle, batch_size)
    while True:
        yield K.batch_get_value(batch)


def input_func(images, labels, shuffle=True, batch_size=128):

    def map_func(image, label):
        img = cv2.imread(image)
        height, width, _ = img.shape

        output_width  = cfg.IMG_INPUT_SIZE / cfg.SHRINK_FACTOR
        output_height = cfg.IMG_INPUT_SIZE / cfg.SHRINK_FACTOR

        x_train = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y_train = np.zeros((output_height, output_width,  cfg.N_ANCHORS, 5 + cfg.N_CLASSES))

        for obj in label:
            xc, yc, w, h, label_idx = obj  # convert label to int
            one_hot_encoding = np.eye(cfg.N_CLASSES)[label_idx]

            # convert to relative value. A cell in grid map
            gt_label = np.concatenate([[xc/float(width), yc/float(height),
                                        w/float(width), h/float(height)],
                                       [1.0],
                                       one_hot_encoding])

            # @TODO: this can be done in loss function or SparseTensor
            center_x = xc * output_width
            center_y = yc * output_height
            r = int(np.floor(center_x))
            c = int(np.floor(center_y))

            # Construct Feature map ground truth
            if r < output_width and c < output_height:
                y_train[c, r, :, :] = cfg.N_ANCHORS * [gt_label]

        return [tf.cast(x_train, tf.float32), tf.cast(y_train, tf.float32)]

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(map_func)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    # This is tensors
    inputs, target = iterator.get_next()

    return inputs, target


if __name__ == "__main__":
    _main_()
