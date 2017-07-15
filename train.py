import keras

from keras.optimizers import Adam
from utils.parse_input import load_data    # Data handler for LISA data set
from yolov2.model import YOLOv2, darknet19
from yolov2.loss  import custom_loss
from utils.data_generator import flow_from_list
from utils.multi_gpu import make_parallel, get_gpus
from cfg import *

# LOAD DATA
lisa_path        = "/home/ubuntu/dataset/training/" # Remember the `/` at the end
pretrained_path  = "/home/ubuntu/dataset/darknet19_544.weights"

x_train, y_train = load_data('training.txt')
labels           = np.unique(y_train[:, 1])
num_classes      = len(labels)            # Count number of classes in the data set
train_data_gen   = flow_from_list(x_train, y_train, ANCHORS, 
                                  batch_size=BATCH_SIZE, 
                                  augment_data=True)
print("Train: {} samples\nNumber of classes: {}".format(len(x_train), num_classes))
print("\n\nAnchors using K-mean clustering [K=5]\n {}".format(ANCHORS))


# CONSTRUCT MODEL
darknet19 = darknet19(pretrained_path, freeze_layers=True)
yolov2    = YOLOv2(feature_extractor=darknet19, num_anchors=len(ANCHORS), num_classes=N_CLASSES)
model     = yolov2.model
model.summary()
# LOAD PRE-TRAINED MODEL
# model.load_weights('/home/ubuntu/dataset/yolov2.weights')

# TRAIN ON MULTI-GPUS
n_gpus = get_gpus()
if n_gpus > 1:
    BATCH_SIZE = n_gpus * BATCH_SIZE
    model = make_parallel(model, n_gpus)

model.compile(optimizer=Adam(LEARN_RATE), loss=custom_loss)


# TRAINING
tf_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
check_pt = keras.callbacks.ModelCheckpoint('models/weights.{epoch:02d}-{loss:.2f}.hdf5', verbose=0, save_best_only=False,
                                           save_weights_only=True, mode='auto', period=1)
hist = model.fit_generator(generator=train_data_gen,
                           steps_per_epoch=3*len(x_train) / BATCH_SIZE,
                           epochs=EPOCHS,
                           callbacks=[tf_board, check_pt],
                           workers=1, verbose=1,
                           initial_epoch=0)
model.save_weights('yolov2.weights')
