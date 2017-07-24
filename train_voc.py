import sys
import os
import keras
import numpy as np
import keras.backend as K
from keras.optimizers import SGD

from cfg import *
from sklearn.utils import shuffle
from utils.parse_input import load_data    # Data handler for LISA dataset
from model.mobilenet import MobileNet
from model.mobile_yolo import MobileYolo
from model.loss import custom_loss, avg_iou
from utils.data_generator import flow_from_list


K.clear_session()  # Avoid duplicate model
# # HYPER-PARAMETERS
BATCH_SIZE = 8
EPOCHS     = 1
LEARN_RATE = 0.00001  # this model has been pre-trained, LOWER LR is needed


print(sys.version)  # Check Python Version

x_train, y_train = load_data('/home/ubuntu/dataset/pascal/train.all.txt')
labels           = np.unique(y_train[:, 1])
num_classes      = len(labels)            # Count number of classes in the dataset

print("\n\nTraining data: {} samples\nNumber of classes: {}".format(len(x_train),num_classes))
print("Label Sample: \n{}".format(y_train[0]))


mobile_net = MobileNet(include_top=False, weights='imagenet')
yolov2     = MobileYolo(feature_extractor=mobile_net, num_anchors=5, num_classes=31, fine_grain_layer=-15)
yolov2.model.summary()

for layer in mobile_net.layers:
    layer.trainable = False

# ### DATA PREPARATION
x_train, y_train = shuffle(x_train, y_train)
train_data_gen = flow_from_list(x_train, y_train, batch_size=BATCH_SIZE, augment_data=True)
x_test, y_test = shuffle(x_train, y_train)[0:600]
val_data_gen = flow_from_list(x_test, y_test, batch_size=BATCH_SIZE, augment_data=False)

tf_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
save_model = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: model.save_weights(
        '/home/ubuntu/dataset/backup/yolov2-epoch%s-loss{%s}.weights' % (epoch, str(logs.get('loss')))))

sgd = SGD(LEARN_RATE, decay=0.0005, momentum=0.9)
adam = keras.optimizers.Adam(LEARN_RATE)


def schedule(epochs):
    if epochs < 5:
        return LEARN_RATE
    if 5 <= epochs < 90:
        return LEARN_RATE * 10.
    if 90 <= epochs <= 120:
        return LEARN_RATE / 10.
    if epochs >= 120:
        return LEARN_RATE / 100.

# Compiling model
lr_scheduler = keras.callbacks.LearningRateScheduler(schedule)
model = yolov2.model
model.compile(optimizer=adam, loss=custom_loss, metrics=[avg_iou])
hist = model.fit_generator(generator=train_data_gen,
                           steps_per_epoch=200,
                           validation_data=val_data_gen,
                           validation_steps=1,
                           epochs=160,
                           callbacks=[tf_board, lr_scheduler, save_model],
                           workers=1, verbose=1)

model.save_weights('yolov2.weights')
