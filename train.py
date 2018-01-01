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
import yaml
import itertools
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K

from yolov2.zoo import yolov2_darknet19
from yolov2.core.loss import yolov2_loss

from yolov2.utils.callbacks import create_callbacks
from yolov2.utils.parser import parse_inputs, parse_label_map
from sklearn.model_selection import train_test_split

from argparse import ArgumentParser

parser = ArgumentParser(description="Train YOLOv2")
parser.add_argument('--csv_file', type=str, default=None,
                    help="Path to CSV training data set")

parser.add_argument('--weights', type=str, default=None,
                    help="Path to pre-trained weight file")


def _main_():
    # ###############
    # PARSE CONFIG  #
    # ###############
    args          = parser.parse_args()

    with open('config.yml', 'r') as stream:
        cfg = yaml.load(stream)

    training_cfg = cfg['training_params']
    training_path = args.csv_file

    weight_file   = cfg['model']['weight_file']
    num_classes   = cfg['model']['num_classes']
    image_size    = cfg['model']['image_size']
    epochs        = training_cfg['epochs']
    batch_size    = training_cfg['batch_size']
    learning_rate = training_cfg['learning_rate']  # this model has been pre-trained, LOWER learning rate is needed
    backup_dir    = training_cfg['backup_dir']

    # Config Anchors and encoding dict
    anchors    = np.array(cfg['anchors'])
    label_dict = parse_label_map(cfg['label_map'])

    # ###############
    # PREPARE DATA  #
    # ###############
    # @TODO: use TFRecords ?
    # @TODO: Create K-fold cross validation training procedure
    inv_map = {v: k for k, v in label_dict.iteritems()}
    inputs, labels = parse_inputs(training_path, inv_map)

    # ###################
    # Define Keras Model
    # ###################
    model = yolov2_darknet19(is_training= True,
                             img_size   = image_size,
                             anchors    = anchors,
                             num_classes= num_classes)

    if weight_file:
        model.load_weights(weight_file)
        print("Weight file has been loaded in to model")

    # ###################
    # COMPILE AND TRAIN #
    # ###################
    model.compile(optimizer= keras.optimizers.Adam(lr=learning_rate),
                  loss     = yolov2_loss(anchors, num_classes))

    for current_epoch in range(epochs):
        # Create 10-fold split
        x_train, x_val = train_test_split(inputs, test_size=0.2)
        y_train = [labels[k] for k in x_train]
        y_val   = [labels[k] for k in x_val]

        print("Number of training samples: {} || {}".format(len(x_train), len(y_train)))
        print("Number of validation samples: {} || {}".format(len(x_val), len(y_val)))

        history = model.fit_generator(generator       = data_generator(x_train, y_train),
                                      steps_per_epoch = 1000,
                                      validation_data = data_generator(x_val, y_val),
                                      validation_steps= 100,
                                      verbose=1,
                                      workers=0)

        # Evaluate y_val and save summaries to tensorboard

    model.save_weights('trained_model.h5')


def data_generator(images, labels, shuffle=True, batch_size=4):
    dataset = input_func(images, labels, shuffle, batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    while True:
        yield K.get_session().run(next_batch)


def input_func(images, labels, shuffle=True, batch_size=8):
    def read_img_file(filename, label):
        image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        return image, label

    def process_label(img, label):
        height = tf.shape(img)[0]
        width = tf.shape(img)[1]
        label = tf.reshape(label, (-1, 5))
        boxes, classes = tf.split(label, [4, 1], 1)
        boxes = boxes / tf.cast([[height, width, height, width]], tf.float32)
        boxes   = tf.expand_dims(boxes, 0)
        img     = tf.expand_dims(img, 0)
        classes = tf.squeeze(classes, axis=1)

        return img, boxes, classes

    # a hack to handle list with diffrent element size.
    dataset = tf.data.Dataset.from_generator(lambda: itertools.izip_longest(images, labels),
                                             (tf.string, tf.float32),
                                             (tf.TensorShape([]), tf.TensorShape([None])))
    dataset = dataset.map(lambda filename, label:
                          tuple(tf.py_func(read_img_file,
                                           [filename, label],
                                           [tf.uint8, label.dtype])))
    dataset = dataset.map(process_label)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=300)

    return dataset


if __name__ == "__main__":
    _main_()
