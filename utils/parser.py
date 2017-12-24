import csv
from itertools import islice
from utils.box import Box, convert_bbox


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
            values: a list of objects in each image in format [xc, xc, w, h, encoded_label]
            e.g dict[image_path] = list of obsjects in that image

    """
    # Extract bounding boxes from training data
    training_instances = {}
    with open(filename, "rb") as f:
        reader = csv.reader(f)
        for line in islice(reader, 1, None):
            if not line:  # Empty line
                continue
            img_path       = line[0]
            cls_name       = line[-1]
            x1, y1, x2, y2 = [float(x) for x in line[1:-1]]
            xc, yc, w, h   = convert_bbox(x1, y1, x2, y2)
            an_object      = [xc, yc, float(w), float(h), label_dict[cls_name]]

            if img_path in training_instances:
                training_instances[img_path].append(an_object)
            else:
                training_instances[img_path] = [an_object]

    return training_instances
