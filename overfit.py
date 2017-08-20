"""
Overfit one image with 1000 epochs to test the loss function properly
"""
import random
import keras

from cfg import *
from utils.parser import parse_inputs
from utils.data_generator import flow_from_list

from models.yolov2 import YOLOv2
from models.yolov2_loss import custom_loss

from argparse import ArgumentParser

parser = ArgumentParser(description="Over-fit one sample to validate YOLOv2 Loss Function")
parser.add_argument('-p', '--path',    help="Path to training text file ", type=str,  default='./dataset/lisa_extension/training.txt')
parser.add_argument('-w', '--weights', help="Path to pre-trained weight files", type=str, default=None)

parser.add_argument('-lr','--learning_rate', type=float, default=0.001)
parser.add_argument('-e', '--epochs',  help='Number of epochs for training', type=int, default=1000)
parser.add_argument('-b', '--batch',   help='Number of batch size', type=int, default=1)

args = parser.parse_args()
annotation_path = args.path
WEIGHTS_FILE    = args.weights
BATCH_SIZE      = args.batch
EPOCHS          = args.epochs
LEARNING_RATE   = args.learning_rate  # this model has been pre-trained, LOWER LR is needed


def _main_():

    # Build Model
    yolov2 = YOLOv2(img_size=(IMG_INPUT, IMG_INPUT, 3), num_classes=N_CLASSES, num_anchors=N_ANCHORS,
                    kernel_regularizer=keras.regularizers.l2(5e-7))

    # Load pre-trained weight if one is available
    #
    for layer in yolov2.layers[:-1]:
        layer.trainable = False

    if WEIGHTS_FILE:
        yolov2.load_weights(WEIGHTS_FILE, by_name=True)

    yolov2.summary()
    # Extract categories
    data = parse_inputs(annotation_path)
    shuffled_keys  = random.sample(data.keys(), len(data.keys()))
    training_dict  = dict([(key, data[key]) for key in shuffled_keys])

    # Create one instance for over-fitting model
    training_dict = dict(training_dict.items()[0:128])
    with open("test_images.csv", "wb") as csv_file:
        fieldnames = ['Filename', 'annotation tag', 'x1', 'y1', 'x2', 'y2']
        import csv
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for fname in training_dict:
            gts = training_dict[fname]
            for gt in gts:
                box, label = gt
                xc, yc, w, h = box.to_array()
                x1 = xc - 0.5*w
                y1 = yc - 0.5*h
                x2 = xc + 0.5*w
                y2 = yc + 0.5*h
                box = "%s, %s, %s, %s"%(x1, y1, x2, y2)
                writer.writerow({'Filename': fname, 'annotation tag': label, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
                print("{}, {}, {}\n".format(fname, box, label))

    # Start training here
    print("Starting training process\n")
    print("Hyper-parameters: LR {} | Batch {} | Optimizers {} | L2 {}".format(LEARNING_RATE, BATCH_SIZE, "SGD", "None"))

    print("Stage 1 Training...Frozen all layers except last one")
    model = yolov2
    model.compile(optimizer=keras.optimizers.adam(lr=0.001), loss=custom_loss)

    train_data_gen = flow_from_list(training_dict, batch_size=16)
    model.fit_generator(generator=train_data_gen, steps_per_epoch=len(training_dict) / 2, epochs=10, workers=3,
                        verbose=1)
    model.save_weights('stage1.weights')

    print("Stage 2 Training...Full training")

    for layer in yolov2.layers:
        layer.trainable = True
    yolov2.load_weights('stage1.weights')

    model = yolov2
    model.compile(keras.optimizers.Adam(lr=0.000002), loss=custom_loss)
    train_data_gen = flow_from_list(training_dict, batch_size=4)
    model.fit_generator(generator=train_data_gen, steps_per_epoch=len(training_dict), epochs=EPOCHS, workers=3,
                        verbose=1)

    model.save_weights('overfit.weights')

if __name__ == "__main__":
    _main_()

    # with open("test_images.csv", "wb") as csv_file:
    #     fieldnames = ['Filename', 'annotation tag', 'x1', 'y1', 'x2', 'y2']
    #     import csv
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     for fname in training_dict:
    #         gts = training_dict[fname]
    #         for gt in gts:
    #             box, label = gt
    #             xc, yc, w, h = box.to_array()
    #             x1 = xc - 0.5*w
    #             y1 = yc - 0.5*h
    #             x2 = xc + 0.5*w
    #             y2 = yc + 0.5*h
    #             box = "%s, %s, %s, %s"%(x1, y1, x2, y2)
    #             writer.writerow({'Filename': fname, 'annotation tag': label, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    #             print("{}, {}, {}\n".format(fname, box, label))
