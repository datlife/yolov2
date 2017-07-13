"""

"""
import numpy as np
from utils.box import Box, convert_bbox


def load_data(txt_file):
    """
    Parse input file into X, Y
    Parameters
    ----------
    input file : text file
            image_path, x1, x2, y1, y2, label1
            image_path, x1, x2, y1, y2, label2
            
            
    img_size (width, height, channels)
    :return: 
            X:  list of image_path
            Y:  list of labels as [x1, x2, y1, y2, label]
    """
    # Extract bounding boxes from training data
    image_paths = []
    gt_boxes    = []
    with open(txt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            img_path, x1, y1, x2, y2, label = line.rstrip().split(",")
            xc, yc, w, h = convert_bbox(x1, y1, x2, y2)
            # xc, yc, w, h = scale_rel_box(img_size, Box(xc, yc, w, h))
            image_paths.append(img_path)
            gt_boxes.append([Box(xc, yc, float(w), float(h)), label])

    print("Number of ground truth boxes: {} boxes".format(len(gt_boxes)))
    return image_paths, np.asarray(gt_boxes)
