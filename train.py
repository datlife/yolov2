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
import itertools
import config as cfg
import keras
import tensorflow as tf
import keras.backend as K

from yolov2.zoo import yolov2_darknet19
from yolov2.core.loss import yolov2_loss

from yolov2.utils.callbacks import create_callbacks
from yolov2.utils.parser import parse_config, parse_inputs
from sklearn.model_selection import train_test_split

from argparse import ArgumentParser

parser = ArgumentParser(description="Train YOLOv2")
parser.add_argument('--csv_file', type=str, default=None,
                    help="Path to CSV training data set")

parser.add_argument('--weights', type=str, default=None,
                    help="Path to pre-trained weight file")

parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs for training')

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
    inv_map = {v: k for k, v in label_dict.iteritems()}
    inputs, labels = parse_inputs(training_path, inv_map)

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

    model.save_weights('trained_model.h5')


def data_generator(images, labels, shuffle=True, batch_size=4):
    dataset    = input_func(images, labels, shuffle, batch_size)
    iterator   = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    while True:
        yield K.get_session().run(next_batch)


def input_func(images, labels, shuffle=True, batch_size=8):

    def read_img_file(filename, label):
        image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (300, 300))
        return image, label

    def process_label(img, label):
        label = tf.reshape(label, (-1, 5))
        gt_boxes, gt_classes = tf.split(label, [4, 1], 1)
        return img, gt_boxes, tf.squeeze(gt_classes, axis=1)

    dataset = tf.data.Dataset.from_generator(lambda: itertools.izip_longest(images, labels),
                                             (tf.string, tf.float32),
                                             (tf.TensorShape([]), tf.TensorShape([None])))
    dataset = dataset.map(lambda filename, label:
                          tuple(tf.py_func(read_img_file,
                                           [filename, label],
                                           [tf.uint8, label.dtype])))
    dataset = dataset.map(process_label)
    if shuffle:
        dataset = dataset.shuffle()

    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == "__main__":
    _main_()
