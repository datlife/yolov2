"""
Data Handler for Training Deep Learning Model
"""
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from utils.box import Box, convert_bbox
from utils.image_handler import random_transform, preprocess_img
from cfg import *


def flow_from_list(x, y, batch_size=32, scaling_factor=5, augment_data=True):
    """
    A ImageGenerator from image paths and return (images, labels) by batch_size

    Parameters
    ---------
    :param x: list of image paths 
    :param y: list of labels as [Box, label_name]

    :param scaling_factor: the level of augmentation. The higher, the more data being augmented
    :param batch_size:     number of images yielded every iteration
    :param augment_data:   enable data augmentation

    Return
    ------
    :return: 
        generate (images, labels) in batch_size
    """
    # @TODO: thread-safe generator (to allow nb_workers > 1)
    slices = int(len(x) / batch_size)
    if augment_data is True:
        augment_level = calc_augment_level(y, scaling_factor)  # (less data / class means more augmentation)
    categories = np.unique(y[:, 1])
    while True:
        x, y = shuffle(x, y)  # Shuffle DATA to avoid over-fitting
        for i in list(range(slices)):
            f_name = x[i * batch_size:(i * batch_size) + batch_size]
            labels = y[i * batch_size:(i * batch_size) + batch_size]
            X = []
            Y = []
            for filename, label in list(zip(f_name, labels)):
                bbox, label = label

                if not os.path.isfile(filename):
                    print('Image Not Found')
                    continue
                img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                height, width, _ = img.shape

                # @TODO multi-scale training
                processed_img = preprocess_img(img)

                # convert label to int
                index_label = np.where(categories == label)[0][0]
                one_hot = np.eye(len(categories))[index_label]

                # convert to relative
                box = bbox.to_relative_size((width, height))
                X.append(processed_img)
                Y.append(np.concatenate([np.array(box), [0.0], one_hot]))

                if augment_data is True:
                    aug_level = augment_level.loc[augment_level['label'] == label, 'scaling_factor'].values[0]
                    for l in list(range(aug_level)):
                        # Create new image & bounding box
                        aug_img, aug_box = random_transform(img, bbox.to_opencv_format())

                        # if box is out-of-bound. skip to next image
                        p1 = (np.asarray([width, height]) - aug_box[0][0])
                        p2 = (np.asarray([width, height]) - aug_box[0][1])
                        if np.any(p1 < 0) or np.any(p2 < 0):
                            continue

                        processed_img = preprocess_img(aug_img)
                        aug_box = convert_opencv_to_box(aug_box)
                        aug_box = aug_box.to_relative_size((width, height))

                        X.append(processed_img)
                        Y.append(np.asarray(np.concatenate([np.array(aug_box), [0.0], one_hot])))

            # Shuffle X, Y again
            X, Y = shuffle(np.array(X), np.array(Y))
            for z in list(range(int(len(X) / batch_size))):
                grid_w = width / SHRINK_FACTOR
                grid_h = height / SHRINK_FACTOR

                # Construct detection mask
                y_batch = np.zeros((batch_size, grid_w, grid_h, N_ANCHORS, 5 + N_CLASSES))

                labels = Y[z * batch_size:(z * batch_size) + batch_size]
                for b in range(batch_size):
                    center_x = labels[b][..., 0] / (float(width) / grid_w)
                    center_y = labels[b][..., 1] / (float(height) / grid_h)

                    c = int(np.floor(center_x))
                    r = int(np.floor(center_y))

                    if r < grid_w and c < grid_h:
                        y_batch[b, r, c, :, 0:4] = N_ANCHORS * [labels[b][..., :4]]
                        y_batch[b, r, c, :, 4] = N_ANCHORS * [1.0]
                        y_batch[b, r, c, :, 5:] = N_ANCHORS * [labels[b][..., 5:]]
                yield X[z * batch_size:(z * batch_size) + batch_size], y_batch.reshape([batch_size, grid_h * grid_w, N_ANCHORS, N_CLASSES + 5])


def calc_augment_level(y, scaling_factor=5):
    """
    Calculate scale factor for each class in data set
    :param y:              List of labels data
    :param scaling_factor: how much we would like to augment each class in data set
    :return: 
    """
    categories, frequencies = np.unique(y[:,1], return_counts=True)  # Calculate how many images in one traffic sign
    mean = frequencies.mean(axis=0)  # average images per traffic sign

    df = pd.DataFrame({'label': categories, 'frequency': frequencies})
    df['scaling_factor'] = df.apply(lambda row: int(scaling_factor*(mean / row['frequency'])), axis=1)
    return df


def convert_opencv_to_box(box):
    x1, y1, x2, y2 = np.array(box).ravel()
    xc, yc, w, h = convert_bbox(x1, y1, x2, y2)
    bbox = Box(xc, yc, w, h)
    return bbox
