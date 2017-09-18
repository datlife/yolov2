"""
This script will create a custom dataset for training and evaluation. It will generate:

   + a CSV training file
   + a CSV validation file (optional: if split is enabled)
   + a categories text file to map indices to label names.
   + an anchor text file (depending on number of anchors, default = 5)

Requirement:
   + a text file, containing ground truths, in this following format:
        path/to/image , x1, y1, x2, y2, label

Example:
-------
python create_custom_dataset.py
 --path       /path/to/text_file.txt
 --output_dir ./dataset/my_new_dataset
 --num_anchors   5
 --split       false

Return
------
  yolov2
  |- dataset
     | - my_new_data_set
         | --  categories.txt
         | --  anchors.txt
         | --  training_data.csv
         | --  testing_data.csv   # if split is enabled
"""
import os
import sys
import csv
from PIL import Image
from sklearn.model_selection import train_test_split

from argparse import ArgumentParser

parser = ArgumentParser(description="Generate custom dataset for training")

parser.add_argument('-p', '--path',
                    help="Path to text file", type=str, default=None)

parser.add_argument('-o', '--output_dir',
                    help="Path to output directory", type=str, default=None)

parser.add_argument('-n', '--number_anchors',
                    help="Number of anchors [default = 5]", type=int, default=5)

parser.add_argument('-s', '--split',
                    help='Splitting data into training/validation set at ratio 0.8/0.2', type=bool, default=False)


def main():
  arguments = parser.parse_args()
  path           = arguments.path
  output_dir     = arguments.output_dir
  number_anchors = arguments.number_anchors
  split          = arguments.split

  # #################################
  # Generate Anchors and Categories #
  # #################################
  with open(path, 'r') as f:
    gt_boxes       = []
    categories     = {}

    lines = f.readlines()
    print("Calculating Anchors....")
    for line in lines:
        img_path, x1, y1, x2, y2, label = line.rstrip().split(",")

        xc, yc, w, h = convert_edges_to_centroid(x1, y1, x2, y2)
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        # aspect_ratio = [IMG_INPUT / float(img_width), IMG_INPUT / float(img_height)]
        # box = Box(0, 0, float(w) * aspect_ratio[0] / SHRINK_FACTOR, float(h) * aspect_ratio[1] / SHRINK_FACTOR)
        gt_boxes.append(box)
    print("Done!")

  categories_file = os.path.join(output_dir, 'categories.txt')
  anchors_file    = os.path.join(output_dir, 'anchors.txt')

  # ###################################
  # Generate Training/Validation data #
  # ###################################
  if split is True:
    with open(path) as txt_file:
      x = txt_file.read().splitlines()[1:]
      x_train, x_test = train_test_split(x, test_size=0.2)

    training_path = os.path.join(output_dir, 'training_data.csv')
    testing_path  = os.path.join(output_dir, 'validation_data.csv')

    save_dataset(x_train, training_path)
    save_dataset(x_test, testing_path)
  else:
    with open(path) as txt_file:
      data = txt_file.read().splitlines()
      save_dataset(data, 'training_data.csv')


def convert_edges_to_centroid(x1, y1, x2, y2):
  w  = float(x2) - float(x1)
  h  = float(y2) - float(y1)
  xc = float(x1) + w / 2.
  yc = float(y1) + h / 2.

  return xc, yc, w, h


def save_dataset(data, save_path):
  with open(save_path, 'w') as txt:
    writer = csv.writer(txt, delimiter=',', )
    writer.writerow(["Filename", "x1", "y1", "x2", "y2", "annotation tag"])
    for item in data:
      item = item.split(',')
      writer.writerow(item)


if __name__ == '__main__':
  main()
