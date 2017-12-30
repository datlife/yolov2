import re
import csv

import numpy as np
from itertools import islice


def parse_inputs(filename, label_dict):
    """
    Read input file and convert into inputs, labels for training

    Parameters
    ----------
    filename : text file
            image_path, x1, x2, y1, y2, label1
            image_path, x1, x2, y1, y2, label2

    label_dict : an encoding dictionary mapping class names to indinces

    Returns:
    --------
    training_instance - a dictionary mapping img path to list of objects
            keys: a list of image paths
            values: a list of objects in each image in format [x1, y1, x2, y2, encoded_label]
            e.g dict[image_path] = list of obsjects in that image

    """
    training_instances = dict()
    with open(filename, "rb") as f:
        reader = csv.reader(f)
        for line in islice(reader, 1, None):
            if not line:
                continue  # Ignore empty line

            img_path       = line[0]
            cls_name       = line[-1]
            x1, y1, x2, y2 = [float(x) for x in line[1:-1]]
            an_object      = [x1, y1, x2, y2, label_dict[cls_name]]

            if img_path in training_instances:
                training_instances[img_path].append(an_object)
            else:
                training_instances[img_path] = [an_object]

    inputs  = training_instances.keys()
    labels  = training_instances

    return inputs, labels


def parse_config(cfg):
    # Config Anchors
    anchors = []
    with open(cfg.ANCHORS, 'r') as f:
        data = f.read().splitlines()
        for line in data:
            numbers = re.findall('\d+.\d+', line)
            anchors.append((float(numbers[0]), float(numbers[1])))

    # Load class names
    with open(cfg.CATEGORIES, mode='r') as txt_file:
        class_names = [c.strip() for c in txt_file.readlines()]

    label_dict = {v: k for v, k in enumerate(class_names)}
    return np.array(anchors), label_dict
