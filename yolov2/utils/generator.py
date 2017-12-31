import cv2
import numpy as np

import tensorflow as tf
import keras.backend as K


def input_func(images, labels, shuffle=True, batch_size=128):
    def map_func(image, label):
        img = cv2.imread(image)
        height, width, _ = img.shape

        output_width = cfg.IMG_INPUT_SIZE / cfg.SHRINK_FACTOR
        output_height = cfg.IMG_INPUT_SIZE / cfg.SHRINK_FACTOR

        x_train = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y_train = np.zeros((output_height, output_width, cfg.N_ANCHORS, 5 + cfg.N_CLASSES))

        for obj in label:
            xc, yc, w, h, label_idx = obj  # convert label to int
            one_hot_encoding = np.eye(cfg.N_CLASSES)[label_idx]

            # convert to relative value
            xc, yc, w, h = bbox.to_relative_size((float(width), float(height)))

            # A cell in grid map
            gt_label = np.concatenate([[xc, yc, w, h], [1.0], one_hot_encoding])
            label.append(gt_label)
            # @TODO: this can be done in loss function
            center_x = xc * output_width
            center_y = yc * output_height
            r = int(np.floor(center_x))
            c = int(np.floor(center_y))

            # Construct Feature map ground truth
            if r < output_width and c < output_height:
                y_train[c, r, :, :] = cfg.N_ANCHORS * [gt_label]

        return [tf.cast(x_train, tf.float32), tf.cast(y_train, tf.float32)]

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(map_func)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    # This is tensors
    inputs, target = iterator.get_next()
    return {'images': inputs}, target


class DataGenerator(Sequence):
    """
    A Keras-way to properly handling multiprocessing dataset
    https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py
    """
    def __init__(self, x, y, config, batch_size, augment=False):
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
        self.cfg                 = config
        self.batch_size          = batch_size
        self.enable_augmentation = augment

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = [self.y[key] for key in batch_x]

        images, labels = self._generate_batch(batch_x, batch_y)

        return np.asarray(images), labels

    def _generate_batch(self, x, y):
        # @TODO: freaking delete this

        """
        Format labels into proper form for computing loss

        :param y: a list of labels : shape (batch_size, objs)
                - each label is a list of objects in an image

        :return:
        """
        images = []
        labels = []

        for filename, ground_truths in zip(x, y):
            image = cv2.imread(filename)
            height, width, _ = image.shape

            label = []

            for obj in ground_truths:
                xc, yc, w, h, label_idx = obj  # convert label to int
                one_hot_encoding = np.eye(self.cfg.N_CLASSES)[label_idx]

                # convert to relative value
                xc, yc, w, h = bbox.to_relative_size((float(width), float(height)))

                # A cell in grid map
                gt_label  = np.concatenate([[xc, yc, w, h], [1.0], one_hot_encoding])
                label.append(gt_label)
                # @TODO: this can be done in loss function
                # center_x = xc * grid_w
                # center_y = yc * grid_h
                # r = int(np.floor(center_x))
                # c = int(np.floor(center_y))
                #
                # # Construct Feature map ground truth
                # if r < grid_w and c < grid_h:
                #     y_batch[i, c, r, :, :] = self.cfg.N_ANCHORS * [object_mask]
            images.append(image)
            labels.append(label)

        return np.array(images), labels
