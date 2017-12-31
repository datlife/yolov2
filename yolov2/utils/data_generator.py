"""
A Threaded Data Generator for YOLOv2

It will generates two things for every batch:

1) Batch of images (in numpy matrices):
   * Dimension : [bach_size, IMG_INPUT, IMG_INPUT, N_ANCHORS * (N_CLASSES+5)]
   * Contain preprocessed input (normalized, re-sized) images

2) Batch of ground truth labels       :
    * Dimension: [batch_size, (IMG_INPUT / SHRINK_FACTOR), (IMG_INPUT / SHRINK_FACTOR), N_ANCHORS * (N_CLASSES+5)]
    * Each ground truth contain : xc, yc, w, h, to, one_hot_labels

"""
import cv2
import random
import pandas as pd
import numpy as np
import threading

from yolov2.utils.box import Box, convert_bbox
from yolov2.utils import augment_img

from config import *

import numpy as np
import cv2
import copy
from yolov2.utils.box import Box


def augment_img(img, objects):
    aug_img      = np.copy(img)
    copy_objects = copy.deepcopy(objects)
    HEIGHT, WIDTH, c = img.shape
    # scale the image
    scale = np.random.uniform() / 10. + 1.
    aug_img = cv2.resize(aug_img, (0, 0), fx=scale, fy=scale)

    # translate the image
    max_offx = (scale - 1.) * WIDTH
    max_offy = (scale - 1.) * HEIGHT
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    aug_img = aug_img[offy: (offy + HEIGHT), offx: (offx + WIDTH)]

    # # Changing color

    # fix object's position and size
    new_labels = []
    for obj in copy_objects:
        gt_box, gt_label = obj
        xc, yc, w, h = gt_box.to_array()
        xc = int(xc * scale - offx)
        xc = max(min(xc, WIDTH), 0)
        yc = int(yc * scale - offy)
        yc = max(min(yc, HEIGHT), 0)
        new_labels.append([Box(xc, yc, w*scale, h*scale), gt_label])

    return aug_img, new_labels


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def flow_from_list(training_instances, batch_size=32, augmentation=False, use_tree=False):
    """
    :param training_instances:
    :param batch_size:
    :return:
    """
    slices   = int(len(training_instances)/batch_size)

    if use_tree is True:
        hier_tree = SoftMaxTree(tree_file=HIERARCHICAL_TREE_PATH)

    # Shuffle data
    keys = training_instances.keys()
    shuffled_keys = random.sample(keys, len(keys))

    while True:
        for i in list(range(slices)):
            instances = shuffled_keys[i * batch_size:(i * batch_size) + batch_size]
            x_batch = np.zeros((batch_size, IMG_INPUT_SIZE, IMG_INPUT_SIZE, 3))
            y_batch = np.zeros((batch_size, IMG_INPUT_SIZE/SHRINK_FACTOR, IMG_INPUT_SIZE/SHRINK_FACTOR, N_ANCHORS, 5 + N_CLASSES))

            for i, instance in enumerate(instances):
                filename    = instance
                objects     = training_instances[instance]
                try:
                    img = cv2.imread(filename)
                    height, width, _ = img.shape
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except IOError:
                    raise IOError("Check input filename. ")

                grid_w = IMG_INPUT_SIZE / SHRINK_FACTOR
                grid_h = IMG_INPUT_SIZE / SHRINK_FACTOR

                if augmentation:
                    img, objects = augment_img(img, objects)

                aug_img = cv2.resize(img, (IMG_INPUT_SIZE, IMG_INPUT_SIZE))

                x_batch[i] = aug_img
                for obj in objects:
                    bbox, label = obj    # convert label to int
                    index = np.where(CLASSES == label)[0][0]

                    if use_tree is False:
                        one_hot = np.eye(N_CLASSES)[index]
                    else:
                        one_hot = hier_tree.encode_label(index)
                        one_hot = one_hot[1:]

                    # convert to relative value
                    xc, yc, w, h = bbox.to_relative_size((float(width), float(height)))
                    object_mask  = np.concatenate([[xc, yc, w, h], [1.0], one_hot])  # A cell in grid map
                    center_x = xc * grid_w
                    center_y = yc * grid_h
                    r = int(np.floor(center_x))
                    c = int(np.floor(center_y))
                    if r < grid_w and c < grid_h:
                        y_batch[i, c, r, :, :] = N_ANCHORS * [object_mask]    # Construct Feature map ground truth

            yield x_batch, y_batch.reshape([batch_size,  IMG_INPUT_SIZE/SHRINK_FACTOR, IMG_INPUT_SIZE/SHRINK_FACTOR, N_ANCHORS*(5 + N_CLASSES)])


def calc_augment_level(y, scaling_factor=5):
    """
    Calculate scale factor for each class in data set
    :param y:              List of labels data
    :param scaling_factor: how much we would like to augment each class in data set
    :return:
    """
    categories, frequencies = np.unique(y[:, 1], return_counts=True)  # Calculate how many images in one traffic sign
    mean = frequencies.mean(axis=0)  # average images per traffic sign
    df = pd.DataFrame({'label': categories, 'frequency': frequencies})
    df['scaling_factor'] = df.apply(lambda row: int(scaling_factor*(mean / row['frequency'])), axis=1)
    return df


def convert_opencv_to_box(box):
    x1, y1, x2, y2 = np.array(box).ravel()
    xc, yc, w, h = convert_bbox(x1, y1, x2, y2)
    bbox = Box(xc, yc, w, h)
    return bbox
