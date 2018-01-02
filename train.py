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
import yaml
import keras
import numpy as np
from sklearn.model_selection import train_test_split

from yolov2.zoo import yolov2_darknet19
from yolov2.core.loss import yolov2_loss
from yolov2.utils.generator import TFData
from yolov2.utils.callbacks import create_callbacks
from yolov2.utils.parser import parse_inputs, parse_label_map

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

    training_path = args.csv_file

    weight_file   = cfg['model']['weight_file']
    num_classes   = cfg['model']['num_classes']
    image_size    = cfg['model']['image_size']
    shrink_factor = cfg['model']['shrink_factor']

    # Config Anchors and encoding dict
    label_dict = parse_label_map(cfg['label_map'])
    anchors = np.array(cfg['anchors']) * (cfg['model']['image_size'] / 608.)

    # ###############
    # PREPARE DATA  #
    # ###############
    # @TODO: use TFRecords ?
    inv_map = {v: k for k, v in label_dict.iteritems()}
    inputs, labels = parse_inputs(training_path, inv_map)

    tfdata = TFData(num_classes, anchors, shrink_factor)

    # ###################
    # Define Keras Model
    # ###################
    # @TODO: how to automatically load weights ?
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

    # From config file
    training_cfg  = cfg['training_params']
    epochs        = training_cfg['epochs']
    batch_size    = training_cfg['batch_size']
    learning_rate = training_cfg['learning_rate']  # this model has been pre-trained, LOWER learning rate is needed
    backup_dir    = training_cfg['backup_dir']

    model.compile(optimizer= keras.optimizers.Adam(lr=learning_rate),
                  loss     = yolov2_loss(anchors, num_classes))

    for current_epoch in range(epochs):
        # Create 10-fold split
        x_train, x_val = train_test_split(inputs, test_size=0.2)
        y_train = [labels[k] for k in x_train]
        y_val   = [labels[k] for k in x_val]

        print("Number of training samples: {} || {}".format(len(x_train), len(y_train)))
        print("Number of validation samples: {} || {}".format(len(x_val), len(y_val)))

        history = model.fit_generator(generator       = tfdata.generator(x_train, y_train, image_size, batch_size),
                                      steps_per_epoch = 1000,
                                      validation_data = tfdata.generator(x_val, y_val, image_size, batch_size),
                                      validation_steps= 100,
                                      verbose         = 1,
                                      workers         = 0)

        # @TODO: Evaluate y_val and save summaries to TensorBoard

    model.save_weights('trained_model.h5')


if __name__ == "__main__":
    _main_()
