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
import keras
import config as cfg
from argparse import ArgumentParser

from yolov2.zoo import yolov2_darknet19
from yolov2.core.loss import yolov2_loss

from yolov2.utils.parser import parse_config


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

parser.add_argument('--initial_epoch',
                    help='Initial epoch (helpful for restart training)', type=int, default=0)

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
    initial_epoch = args.initial_epoch
    backup_dir  = args.backup_dir

    # Config Anchors and encoding dict
    anchors, label_dict = parse_config(cfg)

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print("A backup directory has been created")

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

    # Set up callbacks
    tf_board = keras.callbacks.TensorBoard(log_dir       = './logs',
                                           histogram_freq= 1,
                                           write_graph   = True,
                                           write_images  = True)

    backup = keras.callbacks.ModelCheckpoint(backup_dir +
                                             "best_%s-{epoch:02d}-{val_loss:.2f}.weights" % 'darknet19',
                                             monitor          = 'val_loss',
                                             save_weights_only= True,
                                             save_best_only   = True)

    # ###################
    # COMPILE AND TRAIN #
    # ###################
    model.compile(optimizer= keras.optimizers.Adam(lr=learning_rate),
                  loss     = yolov2_loss(anchors, cfg.N_CLASSES))

    # ###############
    # PREPARE DATA  #
    # ###############
    # @TODO: Create K-fold cross validation training procedure

    # data   = parse_inputs(training_path, label_dict)
    #
    # # Load X, Y
    # X, Y = data.keys, data

    # model.summary()
    # model.save_weights('yolov2.weights')
    # model.fit_generator()


if __name__ == "__main__":
    _main_()
