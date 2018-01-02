import cv2
import itertools
import numpy as np
import tensorflow as tf
import keras.backend as K
from ..core.ops import iou


class TFData(object):
    """A Data Generator Object using tf.data.Dataset


    """
    def __init__(self, num_classes, anchors, shrink_factor=32):
        """

        :param num_classes:   number of classes - for one-hot encoding
        :param anchors:       RELATIVE anchors (in percentage)
        :param shrink_factor: a factor determine how input is shrinked through
                              feature extractor [default = 32]
        """
        self.num_classes   = num_classes
        self.anchors       = anchors
        self.shrink_factor = shrink_factor

    def generator(self, images, labels, img_size, batch_size=4, shuffle=True):
        dataset    = self.create_tfdata(images, labels, img_size, batch_size, shuffle)
        iterator   = dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()
        while True:
            yield K.get_session().run(next_batch)

    def create_tfdata(self, images, labels, img_size, batch_size=4, shuffle=True):

        anchors   = self.anchors * img_size / self.shrink_factor
        upper_pts = anchors - anchors / 2.0
        lower_pts = anchors + anchors / 2.0

        # for generating ground truth later
        anchors_boxes = np.concatenate([upper_pts, lower_pts], axis=-1).astype(np.float32)[[1, 0, 3, 2]]

        def read_img_file(filename, label):
            image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            image = cv2.resize(image, (img_size, img_size))

            # A label is consist of [y1, x1, y2, x2, class_idx]
            label = np.reshape(label, (-1, 5))

            # Convert coordinates to relative values
            boxes = label[..., 0:4] / np.array([height, width, height, width], np.float32)

            # Adjust boxes to correct ratio (due to distorted image)
            # @TODO: box ratio

            label = np.concatenate([boxes, np.expand_dims(label[..., -1], 1)], axis=-1)
            return image, label

        def process_label(img, label):
            boxes, classes = tf.split(label, [4, 1], 1)

            # Generate feature map using tf.scatter_nd
            # https://www.tensorflow.org/api_docs/python/tf/scatter_nd

            # 1. Determine indices (where to put ground truths in the feature map)
            # two ground truths may be in same cell, so we need to calculate the IoU
            iou_scores   = iou(boxes, anchors_boxes)
            cell_indices = tf.cast(tf.argmax(iou_scores, axis=1), tf.int32)
            cell_indices = tf.expand_dims(cell_indices, axis=-1)

            area     = boxes[..., 2:4] - boxes[..., 0:2]
            centroid = boxes[..., 0:2] + (area / 2.0)
            indices  = tf.cast(tf.floor(centroid * (img_size / self.shrink_factor)), tf.int32)
            indices  = tf.concat([indices, cell_indices], axis=-1)

            # 2. Construct output feature
            objectness = tf.ones_like(classes)
            one_hot    = tf.one_hot(tf.cast(tf.squeeze(classes, axis=1), tf.uint8), self.num_classes)
            values     = tf.concat([boxes, objectness, one_hot], axis=1)

            # 3. Create feature map
            feature_map = tf.scatter_nd(indices, values, shape=[img_size / self.shrink_factor,
                                                                img_size / self.shrink_factor,
                                                                len(anchors),
                                                                5 + self.num_classes])
            return img, feature_map

        dataset = tf.data.Dataset.from_generator(lambda: itertools.izip_longest(images, labels),
                                                 (tf.string, tf.float32),
                                                 (tf.TensorShape([]), tf.TensorShape([None])))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.map(lambda filename, label:
                              tuple(tf.py_func(read_img_file,
                                               [filename, label],
                                               [tf.uint8, label.dtype])))
        dataset = dataset.map(process_label)
        dataset = dataset.batch(batch_size)

        return dataset
