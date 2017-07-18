"""
Training YOLOv2

Example:
--------

"""
import keras
from keras.optimizers import Adam, SGD
from utils.parse_input import load_data    # Data handler for LISA data set
from model.yolov2 import YOLOv2, darknet19
from model.loss import custom_loss
from model.metrics import avg_iou, coor, obj
from utils.data_generator import flow_from_list
from utils.multi_gpu import make_parallel, get_gpus
from cfg import *
from argparse import ArgumentParser

parser = ArgumentParser(description="Train YOLOv2")

parser.add_argument('--path',    help="training data in txt format",type=str,  default='training.txt')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--model',   help="Pretrained Weights", type=str,  default='/home/ubuntu/dataset/darknet19_544.weights')
parser.add_argument('--epochs',  help='Steps of training', type=int, default=10)
parser.add_argument('--batch',   type=int, default=8)
parser.add_argument('--learning_rate','-lr', type=float, default=1e-5)



if __name__ == "__main__":
    args = parser.parse_args()
    BATCH_SIZE      = args.batch
    TRAINING_PATH   = args.path
    LEARNING_RATE   = args.learning_rate
    EPOCHS          = args.epochs
    pretrained_path = args.model
    weights         = args.weights

    x_train, y_train = load_data(TRAINING_PATH)
    labels           = np.unique(y_train[:, 1])
    num_classes      = len(labels)            # Count number of classes in the data set
    print("Train: {} samples\nNumber of classes: {}".format(len(x_train), num_classes))
    print("\n\nAnchors using K-mean clustering [K=5]\n {}".format(ANCHORS))

    # CONSTRUCT MODEL
    darknet19 = darknet19(pretrained_path, freeze_layers=True)
    yolov2    = YOLOv2(feature_extractor=darknet19, num_anchors=len(ANCHORS), num_classes=N_CLASSES)
    model     = yolov2.model
    model.summary()

    # LOAD PRE-TRAINED MODEL
    if weights:
        model.load_weights(weights)

    # TRAIN ON MULTI-GPUS
    n_gpus = get_gpus()
    if n_gpus > 1:
        BATCH_SIZE = n_gpus * BATCH_SIZE
        model_par = make_parallel(model, n_gpus)
    else:
        model_par = model

    # optimizer =Adam(LEARNING_RATE)
    optimizer = SGD(lr=0.00001, decay=0.0005, momentum=0.9)

    model_par.compile(optimizer=optimizer, loss=custom_loss, metrics=[avg_iou, coor])

    train_data_gen = flow_from_list(x_train, y_train, batch_size=BATCH_SIZE, augment_data=True)
    # TRAINING
    tf_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=3, mode='min', verbose=1)
    save_model = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: model.save_weights(
            '/home/ubuntu/dataset/backup/yolov2-epoch%s-loss:%s.weights' % (epoch, str(logs.get('loss')))))

    # @TODO :model checkpoint save single-instance model
    hist = model_par.fit_generator(generator=train_data_gen,
                                   steps_per_epoch=100,
                                   epochs=EPOCHS,
                                   callbacks=[tf_board, early_stop, save_model],
                                   workers=8, verbose=1,
                                   initial_epoch=10)

    model.save_weights('yolov2.weights')
