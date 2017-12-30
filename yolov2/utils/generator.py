import cv2
import math
import numpy as np
from keras.utils import Sequence


class DataGenerator(Sequence):
    """
    A Keras-way to properly handling multiprocessing dataset

    https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py

    """
    def __init__(self, x, y, batch_size, augment=False):
        """
        Constructor
        :param x: - a list of image paths for training
        :param y: - a dictionary containing ground truth objects, whereas:
                    Key : an image path
                    Values: a list of objects appearing in the image in format:
                            [(x1, y1, x2, y2, encoded_idx_label), ....]

        :param batch_size:
        :param augment:
        """
        self.x = x
        self.y = y
        self.batch_size          = batch_size
        self.enable_augmentation = augment

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = [self.y[key] for key in batch_x]

        images = [cv2.imread(filename) for filename in batch_x]
        labels = self._process_labels(batch_y)

        return np.asarray(images), labels

    def _process_labels(self, x, y):
        # @TODO: freaking delete this

        """
        Format labels into proper form for computing loss

        :param y: a list of labels : shape (batch_size, objs)
                - each label is a list of objects in an image

        :return:
        """
        images = []
        labels = []
        for filename, labels in zip(x, y):
            image = cv2.imread(filename)
            height, width, _ = image.shape

            for obj in labels:
                xc, yc, w, h, label_idx = obj  # convert label to int
                one_hot_encoding = np.eye(N_CLASSES)[label_idx]

                # convert to relative value
                xc, yc, w, h = bbox.to_relative_size((float(width), float(height)))

                # A cell in grid map
                object_mask  = np.concatenate([[xc, yc, w, h], [1.0], one_hot])

                center_x = xc * grid_w
                center_y = yc * grid_h
                r = int(np.floor(center_x))
                c = int(np.floor(center_y))

                # Construct Feature map ground truth
                if r < grid_w and c < grid_h:
                    y_batch[i, c, r, :, :] = N_ANCHORS * [object_mask]

        return 0