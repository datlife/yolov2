import re
import csv
from itertools import islice

import numpy as np


def parse_inputs(filename, label_dict):
  """
    Read input file and convert into inputs, labels for training

  Args:
    filename: text file - has content format as
        image_path, x1, x2, y1, y2, label1
        image_path, x1, x2, y1, y2, label2
    label_dict:  an encoding dictionary -
      mapping class names to indices

  Returns:
    inputs :  a list of all image paths to dataset
    labels :  a dictionary,
      key : image_path
      value: all objects in that image

  @TODO: when dataset is large, we should consider this method a generator
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
      an_object = [y1, x1, y2, x2, label_dict[cls_name]]

      if img_path in training_instances:
        training_instances[img_path].append(an_object)
      else:
        training_instances[img_path] = [an_object]
  inputs = training_instances.keys()
  labels = {k: np.stack(v).flatten() for k, v in training_instances.items()}

  return inputs, labels


def parse_label_map(label_map_path):
  """Parse label map file into a dictionary
  Args:
    label_map_path:

  Returns:
    a dictionary : key: obj_id value: obj-name
  """
  # match any group having language of {id:[number] .. name:'name'}
  parser = re.compile(r'id:[^\d]*(?P<id>[0-9]+)\s+name:[^\']*\'(?P<name>[\w_-]+)\'')

  with open(label_map_path, 'r') as f:
    lines = f.read().splitlines()
    lines = ''.join(lines)

    # a tuple (id, name)
    result = parser.findall(lines)
    label_map_dict = {item[0]: item[1] for item in result}

    return label_map_dict

