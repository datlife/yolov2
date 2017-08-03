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
import keras
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import LambdaCallback
from keras.callbacks import LearningRateScheduler

from cfg import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils.parse_txt_to_inputs import parse_txt_to_inputs    # Data handler for LISA dataset
from utils.data_generator import flow_from_list
from utils.multi_gpu import get_gpus, make_parallel

from model.darknet19 import darknet19
from model.YOLOv2 import YOLOv2
from model.loss import custom_loss

from argparse import ArgumentParser
parser = ArgumentParser(description="Train YOLOv2")
parser.add_argument('-p', '--path', help="Path to training data set (e.g. /dataset/lisa/ ", type=str,  default='training.txt')
parser.add_argument('-w', '--weights', help="Path to pre-trained weight files", type=str, default=None)
parser.add_argument('-e', '--epochs',  help='Number of epochs for training', type=int, default=10)
parser.add_argument('-b', '--batch',   help='Number of batch size', type=int, default=4)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-s', '--backup',  help='Path to to backup model directory', type=str, default='/media/sharedHDD/yolo_model/')
parser.add_argument('--feature_extractor_weights', help="Path to feature extractor pre-trained weights", type=str, default=None)

args = parser.parse_args()
annotation_path = args.path
WEIGHTS_FILE    = args.weights
BATCH_SIZE      = args.batch
EPOCHS          = args.epochs
LEARNING_RATE   = args.learning_rate  # this model has been pre-trained, LOWER LR is needed
BACK_UP_PATH    = args.backup
FEATURE_EXTRACTOR_WEIGHTS = args.feature_extractor_weights
K.clear_session()  # Avoid duplicate model


def _main_():
    # Load data
    x_train, y_train = parse_txt_to_inputs(annotation_path)
    # Build Model
    darknet  = darknet19(input_size=(IMG_INPUT, IMG_INPUT, 3), pretrained_weights=FEATURE_EXTRACTOR_WEIGHTS)
    yolov2   = YOLOv2(feature_extractor=darknet, num_anchors=N_ANCHORS, num_classes=N_CLASSES, fine_grain_layer=['leaky_re_lu_13'])
    # Build Model

    # Load pre-trained file if one is available
    if WEIGHTS_FILE:
        yolov2.model.load_weights(WEIGHTS_FILE, by_name=True)

    for layer in darknet.layers:
        layer.trainable = False
    yolov2.model.summary()

    # for Debugging during training
    tf_board, lr_scheduler, backup_model = setup_debugger(yolov2)

    # Update Multiple GPUS- training
    gpus = get_gpus()
    if gpus > 1:
        model = make_parallel(yolov2.model, gpus)
        batch = gpus*BATCH_SIZE
        print("Multiple GPUs training is enabled")

    adam = keras.optimizers.Adam(LEARNING_RATE)
    model.compile(optimizer=adam, loss=custom_loss)

    # Start training here
    # Construct Data Generator
    x, y = shuffle(x_train, y_train)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    train_data_gen = flow_from_list(x_train, y_train, batch_size=batch, augment_data=False)
    val_data_gen = flow_from_list(x_test, y_test, batch_size=batch, augment_data=False)

    print("Starting training process\n")
    model.fit_generator(generator=train_data_gen,
                        steps_per_epoch=int(len(x_train)/BATCH_SIZE),
                        validation_data=val_data_gen,
                        validation_steps=int(len(x_test)/BATCH_SIZE),
                        epochs=EPOCHS, initial_epoch=0,
                        callbacks=[tf_board, lr_scheduler, backup_model],
                        workers=3, verbose=1)

    yolov2.model.save_weights('yolov2.weights')


def create_data_generator(x_train, y_train):
    """
    Create data generator
    :param x_train:
    :param y_train:
    :return:
    """
    x_train, y_train = shuffle(x_train, y_train)
    train_data_gen = flow_from_list(x_train, y_train, batch_size=BATCH_SIZE, augment_data=False, scaling_factor=AUGMENT_LEVEL)
    x_test, y_test = shuffle(x_train, y_train)[0:int(len(x_train)*0.2)]
    val_data_gen = flow_from_list(x_test, y_test, batch_size=BATCH_SIZE, augment_data=False)
    return train_data_gen, val_data_gen


def setup_debugger(yolov2):
    """
    Debugger for monitoring model during training
    :param yolov2:
    :return:
    """
    tf_board     = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    backup_model   = LambdaCallback(on_epoch_end=lambda epoch, logs: yolov2.model.save_weights(BACK_UP_PATH+'yolov2-epoch%s-loss%s.weights'% (epoch, str(logs.get('val_loss')))))
    lr_scheduler = LearningRateScheduler(learning_rate_schedule)
    return tf_board, lr_scheduler, backup_model


def learning_rate_schedule(epochs):
    if epochs < 60:
        return LEARNING_RATE
    if 60 <= epochs < 90:
        return LEARNING_RATE / 10.
    if 90 <= epochs < 140:
        return LEARNING_RATE / 100.
    if epochs >= 140:
        return LEARNING_RATE / 1000.

if __name__ == "__main__":
    _main_()
    # # OVER-FIT ON FEW EXAMPLES
    # len = 8
    # x_train, y_train = shuffle(x_train, y_train)
    # x_train = np.tile(x_train[0:len], 4).tolist()
    # y_train  = np.tile(y_train[0:len], [4, 1])
    # with open("test_images.txt", "w") as text_file:
    #     for fname in x_train[0:len]:
    #         text_file.write("%s\n" % fname)
    #         print(fname)
    #
    # train_data_gen = flow_from_list(x_train, y_train, batch_size=BATCH_SIZE, augment_data=False)
    # val_data_gen = flow_from_list(x_train, y_train,   batch_size=BATCH_SIZE, augment_data=False)
