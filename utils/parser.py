import csv
from itertools import islice

from utils.box import Box, convert_bbox


def parse_inputs(txt_file):
    """
    Parse input file into X, Y
    Parameters
    ----------
    input file : text file
            image_path, x1, x2, y1, y2, label1
            image_path, x1, x2, y1, y2, label2
    img_size (width, height, channels)
    :return: A dictionary
            dict[image_path] = list of obsjects in that image
            X:  list of image_path
            Y:  list of labels as [xc, xc, w, h, label]
    """
    # Extract bounding boxes from training data
    training_instances = {}
    with open(txt_file, "rb") as f:
        reader = csv.reader(f)
        for line in islice(reader, 1, None):
            if not line:  # Empty line
                continue
            img_path = line[0]
            x1, y1, x2, y2 = [float(x) for x in line[1:-1]]
            xc, yc, w, h   = convert_bbox(x1, y1, x2, y2)
            an_object      = [Box(xc, yc, float(w), float(h)), line[-1]]
            if img_path in training_instances:
                training_instances[img_path].append(an_object)
            else:
                training_instances[img_path] = [an_object]

    return training_instances

# Test new parse input
if __name__ == "__main__":
    from scripts.split_dataset import split_data
    training, testing = split_data('../training.txt', ratio=0.2)
    training_data = parse_inputs(training)
    import random
    keys = training_data.keys()
    shuffled_keys = random.sample(keys, len(keys))
    for key in shuffled_keys:
        if len(training_data[key]) > 1:
            print(key, training_data[key])
    print("Total images: ", len(training_data))
