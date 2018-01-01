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

    """
    training_instances = dict()
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for line in islice(reader, 1, None):
            if not line:
                continue  # Ignore empty line

            img_path = line[0]
            cls_name = line[-1]
            x1, y1, x2, y2 = [float(x) for x in line[1:-1]]
            an_object = [y1, x1, y2, x2,  label_dict[cls_name]]

            if img_path in training_instances:
                training_instances[img_path].append(an_object)
            else:
                training_instances[img_path] = [an_object]
    inputs = training_instances.keys()
    labels = {k: np.stack(v).flatten() for k, v in training_instances.iteritems()}

    return inputs, labels


def parse_label_map(map_file):
    try:
        with open(map_file, mode='r') as txt_file:
            class_names = [c.strip() for c in txt_file.readlines()]

        label_dict = {v: k for v, k in enumerate(class_names)}
        return label_dict

    except IOError as e:
        print("\nPlease check config.py file")
        raise(e)


class Dict2Obj(object):
    """ Convert dict to object """
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Dict2Obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Dict2Obj(b) if isinstance(b, dict) else b)
