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


def flow_from_list(x, y, anchors, batch_size=32, scaling_factor=5, augment_data=True):
    """
    A ImageGenerator from image paths and return (images, labels) by batch_size

    Parameters
    ---------
    :param x: list of image paths 
    :param y: list of labels as [Box, label_name]

    :param anchors:        list of anchors
    :param scaling_factor: the level of augmentation. The higher, the more data being augmented
    :param batch_size:     number of images yielded every iteration
    :param augment_data:   enable data augmentation

    Return
    ------
    :return: 
        generate (images, labels) in batch_size
    """
    # Convert to panda frames
    if augment_data is True:
        augment_level = calc_augment_level(y, scaling_factor)  # (less data / class means more augmentation)

    # @TODO: thread-safe generator (to allow nb_workers > 1)
    slices = int(len(x) / batch_size)

    categories = np.unique(y[:, 1])
    label_map = pd.factorize(categories)[0]
    while True:
        x, y = shuffle(x, y)  # Shuffle DATA to avoid over-fitting
        for i in list(range(slices)):
            fnames = x[i * batch_size:(i * batch_size) + batch_size]
            labels = y[i * batch_size:(i * batch_size) + batch_size]
            X = []
            Y = []
            for filename, label in list(zip(fnames, labels)):
                bbox, label = label
                if not os.path.isfile(filename):
                    print('Not Found')
                    continue

                # Determine how much augmentation this image needs
                aug_level = augment_level.loc[augment_level['label'] == label, 'scaling_factor'].values[0]
                img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                height, width, _ = img.shape

                # Prep-rocess image **IMPORTANT
                processed_img = preprocess_img(img)

                # convert label to int
                index_label = np.where(categories == label)[0][0]
                xc, yc, w, h = bbox.to_relative_size(img_size=(width, height))

                X.append(processed_img)
                Y.append([xc, yc, w, h, index_label])

                if augment_data is True:
                    for l in list(range(aug_level)):
                        # Create new image & bounding box
                        aug_img, aug_box = random_transform(img, bbox.to_opencv_format())
                        aug_label = index_label

                        # if box is out-of-bound. skip to next image
                        p1 = (np.asarray([width, height]) - aug_box[0][0])
                        p2 = (np.asarray([width, height]) - aug_box[0][1])
                        if np.any(p1 < 0) or np.any(p2 < 0):
                            continue

                        processed_img = preprocess_img(aug_img)
                        aug_box = convert_opencv_to_box(aug_box)

                        xc, yc, w, h = aug_box.to_relative_size(img_size=(width, height))
                        X.append(processed_img)
                        Y.append([xc, yc, w, h, aug_label])

            # Shuffle X, Y again
            X, Y = shuffle(X, Y)
            Y = np.array(Y)

            # Generate (augmented data + original data) in correct batch_size
            iterations = list(range(int(len(X) / batch_size)))
            for z in iterations:
                yield X[z * batch_size:(z * batch_size) + batch_size], Y[z * batch_size:(z * batch_size) + batch_size]


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

def preprocess_true_boxes(true_boxes, anchors, image_size):
    """Find detector in YOLO where ground truth box should appear.
    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative x, y, w, h, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.

    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.

    image_size : array-like
        List of image dimensions in form of h, w in pixels.

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    height, width, _ = image_size
    num_anchors = len(anchors)

    # TODO: Remove hard-coding of downscaling calculations.
    conv_height = height // 32
    conv_width = width // 32

    num_box_params = true_boxes.shape[1]
    detectors_mask = np.zeros((conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros((conv_height, conv_width, num_anchors, num_box_params), dtype=np.float32)

    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5]
        print(box)
        box = box[0:4] * np.array([conv_width, conv_height, conv_width, conv_height])
        print(box)

        i = np.floor(box[1]).astype('int')
        j = np.floor(box[0]).astype('int')

        best_iou = 0
        best_anchor = 0
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)

            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        if best_iou > 0:
            detectors_mask[i, j, best_anchor] = 1
            adjusted_box = np.array(
                [
                    box[0] - j, box[1] - i,
                    np.log(box[2] / anchors[best_anchor][0]),
                    np.log(box[3] / anchors[best_anchor][1]), box_class
                ],
                dtype=np.float32)
            matching_true_boxes[i, j, best_anchor] = adjusted_box
    return detectors_mask, matching_true_boxes
