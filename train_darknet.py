import keras
import numpy as np
import keras.backend as K
from keras.optimizers import SGD

from cfg import *
from sklearn.utils import shuffle
from utils.parse_input import load_data    # Data handler for LISA dataset
from model.darknet19 import darknet19
from model.mobile_yolo import MobileYolo
from model.loss import custom_loss, avg_iou, recall, precision
from utils.data_generator import flow_from_list
from utils.multi_gpu import make_parallel, get_gpus

K.clear_session()  # Avoid duplicate model
# # HYPER-PARAMETERS
BATCH_SIZE   = 8
EPOCHS       = 1
LEARN_RATE   = 0.001  # this model has been pre-trained, LOWER LR is needed
BACK_UP_PATH = '/home/ubuntu/dataset/backup/yolov2-epoch%s-loss{%s}.weights'

# LOAD DATA
x_train, y_train = load_data('training.txt')
labels           = np.unique(y_train[:, 1])
num_classes      = len(labels)            # Count number of classes in the dataset


darknet    = darknet19(input_size=(None, None, 3), pretrained_weights='../yolo-coco.h5')
yolov2     = MobileYolo(feature_extractor=darknet, num_anchors=5, num_classes=31, fine_grain_layer=43)
yolov2.model.summary()

for layer in darknet.layers:
    layer.trainable = False

# ### DATA PREPARATION
x_train, y_train = shuffle(x_train, y_train)
# train_data_gen = flow_from_list(x_train, y_train, batch_size=BATCH_SIZE, augment_data=True)
# x_test, y_test = shuffle(x_train, y_train)[0:600]
# val_data_gen = flow_from_list(x_test, y_test, batch_size=BATCH_SIZE, augment_data=False)

one_sample = np.tile(x_train[0:30], 2).tolist()
one_label  = np.tile(y_train[0:30], [2, 1])
print([name.split('/')[-1].split('.')[0] for name in one_sample])
print(one_label)
print(one_sample[1])

train_data_gen = flow_from_list(one_sample, one_label, batch_size=BATCH_SIZE, augment_data=True)
val_data_gen = flow_from_list(one_sample, one_label, batch_size=BATCH_SIZE, augment_data=False)

tf_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
save_model = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs:
                                            model.save_weights(BACK_UP_PATH % (epoch, str(logs.get('loss')))))

sgd = SGD(LEARN_RATE, decay=0.0005, momentum=0.9)
adam = keras.optimizers.Adam(LEARN_RATE)


def schedule(epochs):
    if epochs < 90:
        return LEARN_RATE
    if 90 <= epochs < 140:
        return LEARN_RATE / 10.
    if epochs >= 140:
        return LEARN_RATE / 1000.

lr_scheduler = keras.callbacks.LearningRateScheduler(schedule)


# Compiling model
model = yolov2.model
# model.load_weights('yolov2.weights')

model.compile(optimizer=adam, loss=custom_loss, metrics=[avg_iou, precision])
hist = model.fit_generator(generator=train_data_gen,
                           steps_per_epoch=100,
                           validation_data=val_data_gen,
                           validation_steps=1,
                           epochs=30, initial_epoch=0,
                           callbacks=[tf_board, lr_scheduler, save_model],
                           workers=2, verbose=1)

model.save_weights('yolov2.weights')
