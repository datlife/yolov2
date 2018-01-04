import cv2
import itertools
import numpy as np
import tensorflow as tf
import keras.backend as K
from ..core.ops import iou
from ..core.ops import find_and_solve_collided_indices


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
        self.num_classes = num_classes
        self.anchors = anchors
        self.shrink_factor = shrink_factor

    def generator(self, images, labels, img_size, shuffle=True, batch_size=4):
        dataset = self.create_tfdata(images, labels, img_size, shuffle, batch_size)
        iterator = dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()
        while True:
            yield K.get_session().run(next_batch)

    def create_tfdata(self, images, labels, img_size, shuffle=True, batch_size=4):
        # for generating ground truth later
        # swapping width and height

        anchors = np.array(self.anchors).astype(np.float32)[:, [1, 0]]

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
            area = boxes[..., 2:4] - boxes[..., 0:2]
            centroids = boxes[..., 0:2] + (area / 2.0)

            anchor_centroids = tf.tile(centroids, (1, len(anchors)))
            anchor_centroids = tf.reshape(anchor_centroids, shape=[tf.shape(centroids)[0], len(anchors), -1])

            upper_pts = anchor_centroids - (anchors / 2.0)
            lower_pts = anchor_centroids + (anchors / 2.0)
            anchor_boxes = tf.cast(tf.concat([upper_pts, lower_pts], axis=-1), tf.float32)

            iou_scores = tf.map_fn(lambda x: iou(tf.expand_dims(x[0], 0), x[1]),
                                   elems=(boxes, anchor_boxes), dtype=tf.float32)

            z_indices = tf.cast(tf.argmax(tf.squeeze(iou_scores, 1), axis=1), tf.int64)
            z_indices = tf.expand_dims(z_indices, axis=-1)
            xy_indices = tf.cast(tf.floor(centroids * (img_size / self.shrink_factor)), tf.int64)

            # @TODO: solve collided indices, bboxes will be accumulated
            indices = tf.concat([xy_indices, z_indices], axis=-1)

            # 2. Construct output feature
            objectness = tf.ones_like(classes)
            one_hot = tf.one_hot(tf.cast(tf.squeeze(classes, axis=1), tf.uint8), self.num_classes)
            values = tf.concat([boxes, objectness, one_hot], axis=1)

            output_shape = tf.TensorShape([img_size / self.shrink_factor,
                                           img_size / self.shrink_factor,
                                           len(anchors),
                                           5 + self.num_classes])
            # Filter out duplicated indices
            indices, values = tf.cond(pred=tf.greater(tf.shape(indices)[0], 1),
                                      true_fn=lambda: find_and_solve_collided_indices(indices, values, output_shape),
                                      false_fn=lambda: [indices, values])
            # 3. Create feature map
            # @TODO: SparseTensor ?
            # https://www.tensorflow.org/api_docs/python/tf/scatter_n
            feature_map = tf.scatter_nd(indices, values, shape=output_shape)
            # feature_map = tf.SparseTensor(indices, values,output_shape)

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

