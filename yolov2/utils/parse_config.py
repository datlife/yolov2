import re
import numpy as np


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