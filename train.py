"""
Training object detector for YOLOv2 on new dataset.

Assumption:
----------
   * Configuration file `cfg.yaml has been setup properly, including:
         + feature extractor type
         + image size input
         + number of classes
         + path to anchors file, categories

   * A training and validation CSV files in correct format have been generated. (could be done through `create_dataset.py`).

Return
-------


Example:

python train.py \
--train-path    dataset/pascal/training_data.csv \
--config        dataset/pascal/config.yml  \

"""
import yaml
import keras
import numpy as np
from keras.layers import Input
from sklearn.model_selection import train_test_split

from yolov2.zoo import yolov2_darknet19
from yolov2.core.loss import YOLOV2Loss
from yolov2.utils.generator import TFData
from yolov2.utils.callbacks import create_callbacks
from yolov2.utils.parser import parse_inputs, parse_label_map

from argparse import ArgumentParser

parser = ArgumentParser(description="Train YOLOv2")
parser.add_argument('--csv_file', type=str, default=None,
                    help="Path to CSV training data set")
parser.add_argument('--config', type=str, default=None,
                    help="Path to config file")


def _main_():
    # ###############
    # PARSE CONFIG  #
    # ###############
    args = parser.parse_args()
    cfg  = yaml.load(open(args.config, 'r'))

    training_data = args.csv_file
    weight_file   = cfg['model']['weight_file']
    num_classes   = cfg['model']['num_classes']
    image_size    = cfg['model']['image_size']
    shrink_factor = cfg['model']['shrink_factor']

    # Hyper-parameters
    training_cfg  = cfg['training_params']
    epochs        = training_cfg['epochs']
    batch_size    = training_cfg['batch_size']
    learning_rate = training_cfg['learning_rate']  # this model has been pre-trained, LOWER learning rate is needed
    backup_dir    = training_cfg['backup_dir']

    # Config Anchors and encoding dict
    label_dict = parse_label_map(cfg['label_map'])

    # @TODO: remove 608.
    anchors = np.array(cfg['anchors']) / (608. / shrink_factor)

    # ###############
    # PREPARE DATA  #
    # ###############
    inv_map = {v: k for k, v in label_dict.iteritems()}
    images, labels = parse_inputs(training_data, inv_map)

    # using tf.data.Dataset to generate training samples
    tfdata = TFData(num_classes, anchors, shrink_factor)

    # ###################
    # Define Keras Model
    # ###################
    # @TODO: how to automatically load weights ?
    inputs = Input(shape=(None, None, 3))
    model = yolov2_darknet19(inputs,
                             is_training= True,
                             anchors    = anchors,
                             num_classes= num_classes)

    # ###################
    # COMPILE AND TRAIN #
    # ###################
    objective_function = YOLOV2Loss(anchors, num_classes)

    model.compile(optimizer= keras.optimizers.Adam(lr=learning_rate),
                  loss     = objective_function.compute_loss)

    if weight_file:
        model.load_weights(weight_file, by_name=True)
        print("Weight file has been loaded in to model")

    # initialize the weights of the detection (last) layer to avoid nan when just starting training
    layer      = model.layers[-2]
    new_kernel = np.random.normal(size=layer.get_weights()[0].shape) / ((image_size / shrink_factor) ** 2)
    new_bias   = np.random.normal(size=layer.get_weights()[1].shape) / ((image_size / shrink_factor) ** 2)
    layer.set_weights([new_kernel, new_bias])

    model.summary()
    callbacks = create_callbacks(backup_dir)

    for current_epoch in range(epochs):
        # Create 10-fold split
        x_train, x_val = train_test_split(images, test_size=0.2)
        y_train = [labels[k] for k in x_train]
        y_val   = [labels[k] for k in x_val]

        model.fit_generator(generator       = tfdata.generator(x_train, y_train, image_size, batch_size),
                            steps_per_epoch = 1000,
                            validation_data = tfdata.generator(x_val, y_val, image_size, batch_size),
                            validation_steps= int(len(x_val)/ batch_size),
                            callbacks       = callbacks,
                            verbose         = 1,
                            workers         = 0)

        # @TODO: Evaluate y_val and save summaries to TensorBoard

    model.save_weights('trained_model.h5')


if __name__ == "__main__":
    _main_()
