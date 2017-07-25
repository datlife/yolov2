import keras
import numpy as np
import keras.backend as K
from argparse import ArgumentParser

from cfg import *
from sklearn.utils import shuffle
from utils.parse_input import load_data    # Data handler for LISA dataset
from utils.data_generator import flow_from_list

from model.darknet19 import darknet19
from model.mobile_yolo import MobileYolo
from model.loss import custom_loss, avg_iou, precision

K.clear_session()  # Avoid duplicate model

parser = ArgumentParser(description="Train YOLOv2")

parser.add_argument('-p', '--path',    help="training data in txt format", type=str,  default='training.txt')
parser.add_argument('-w', '--weights', type=str, default=None)
parser.add_argument('-e', '--epochs',  help='Steps of training', type=int, default=10)
parser.add_argument('-b', '--batch',   type=int, default=8)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)

args = parser.parse_args()
annotation_path = args.path
WEIGHTS_FILE    = args.weights
BATCH_SIZE      = args.batch
EPOCHS          = args.epochs
LEARNING_RATE   = args.learning_rate  # this model has been pre-trained, LOWER LR is needed
BACK_UP_PATH    = '/media/sharedHDD/yolo_model/yolov2-epoch%s-loss%s.weights'


def learning_rate_schedule(epochs):
    if epochs < 90:
        return LEARNING_RATE
    if 60 <= epochs < 90:
        return LEARNING_RATE / 10.
    if 90 <= epochs < 140:
        return LEARNING_RATE / 100.
    if epochs >= 140:
        return LEARNING_RATE / 1000.


def _main_():

    # Load data
    x_train, y_train = load_data(annotation_path)

    # Build Model
    darknet    = darknet19(input_size=(608, 608, 3), pretrained_weights='yolo-coco.h5')
    yolov2     = MobileYolo(feature_extractor=darknet, num_anchors=5, num_classes=N_CLASSES, fine_grain_layer=43)
    yolov2.model.summary()

    for layer in darknet.layers:
        layer.trainable = False

    # Construct Data Generator
    x_train, y_train = shuffle(x_train, y_train)
    train_data_gen = flow_from_list(x_train, y_train, batch_size=BATCH_SIZE, augment_data=True)
    x_test, y_test = shuffle(x_train, y_train)[0:600]
    val_data_gen = flow_from_list(x_test, y_test, batch_size=BATCH_SIZE, augment_data=False)

    # for Debugging during training
    tf_board     = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    save_model   = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs:
                                                  yolov2.model.save_weights(BACK_UP_PATH % (epoch, str(logs.get('val_loss')))))
    lr_scheduler = keras.callbacks.LearningRateScheduler(learning_rate_schedule)

    adam = keras.optimizers.Adam(LEARNING_RATE)
    if WEIGHTS_FILE:
        yolov2.model.load_weights(WEIGHTS_FILE)
    
    print("Starting training process - Hyperameter Settings:\n")
    print("Learning Rate: {}\nBatch Size: {}\n EPOCHS: {})".format(LEARNING_RATE, BATCH_SIZE, EPOCHS))
    yolov2.model.compile(optimizer=adam, loss=custom_loss, metrics=[avg_iou, precision])
    yolov2.model.fit_generator(generator=train_data_gen,
                               steps_per_epoch=500,
                               validation_data=val_data_gen,
                               validation_steps=1,
                               epochs=200, initial_epoch=0,
                               callbacks=[tf_board, lr_scheduler, save_model],
                               workers=2, verbose=1)

    yolov2.model.save_weights('yolov2.weights')


if __name__ == "__main__":
    _main_()

    # one_sample = np.tile(x_train[0:4], 8).tolist()
    # one_label  = np.tile(y_train[0:4], [8, 1])
    # print([name.split('/')[-1].split('.')[0] for name in one_sample])
    # print(one_sample[1])
    # train_data_gen = flow_from_list(one_sample, one_label, batch_size=BATCH_SIZE, augment_data=True)
    # val_data_gen = flow_from_list(one_sample, one_label,   batch_size=BATCH_SIZE, augment_data=False)
