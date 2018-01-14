"""
This script creates a custom dataset for training and evaluation. It will generate:

   * A CSV training file
   * A categories text file to map indices to label names.
   * A configuration .yml file (with anchors = 5 by default)

Requirement:
-----------
   + a text file, containing ground truths, in following format:
        image.jpg , x1, y1, x2, y2, label
        image2.jpg, x1, y1, x2, y2, label2

    whereas:
       image.jpg      : a str - absolute image path on local host
       x1, y1, x2, y2: float  - absolute object coordinates

       label:          string - name of label

Example usage:
-------------
python create_custom_dataset.py
 --data_file     /path/to/text_file.txt
 --output_dir   ./dataset/new_dataset
 --num_anchors   5

Return
------
  |- dataset
     | - my_new_data_set
         | --  config.py
         | --  categories.txt
         | --  anchors.txt
         | --  training_data.csv
"""
# https://www.tensorflow.org/programmers_guide/datasets

import os
import logging
import csv
import numpy as np
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser(description="Generate custom dataset for training")
parser.add_argument('-p', '--data_file',
                    help="Path to text file", type=str, default=None)

parser.add_argument('-o', '--output_dir',
                    help="Path to output directory", type=str, default=None)

parser.add_argument('-n', '--num_anchors',
                    help="Number of anchors [default = 5]", type=int, default=5)


# @TODO: Stratifield Kfold for splitting dataset
def main():
  arguments = parser.parse_args()
  data_file = arguments.data_file
  output_dir = arguments.output_dir
  num_anchors = arguments.num_anchors

  # #################################
  # Generate Anchors and Categories #
  # #################################
  with open(data_file, 'r') as fio:
    logging.info("Reading data")
    gt_boxes = []
    for line in fio:
      img_path, x1, y1, x2, y2, label = line.rstrip().split(",")
      xc, yc, w, h = convert_edges_to_centroid(x1, y1, x2, y2)
      with Image.open(img_path) as img:
        img_width, img_height = img.size

      # Convert to relative bboxes
      relative_bbox = np.array([xc, yc, w, h])/np.array([img_width, img_height, img_width, img_height], dtype=np.float)
      gt_boxes.append(relative_bbox)

    logging.info("Calculating Anchors using K-mean Clustering....")
    anchors, avg_iou = k_mean_cluster(num_anchors, gt_boxes)
    logging.info("Number of anchors: {:2} | Average IoU:{:-4f}\n\n ".format(num_anchors, avg_iou))

  print(anchors)

  # ###################################
  # Generate Training/Validation data #
  # ###################################


def convert_edges_to_centroid(x1, y1, x2, y2):
  w = float(x2) - float(x1)
  h = float(y2) - float(y1)
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


def k_mean_cluster(n_anchors, gt_boxes, loss_convergence=1e-5):
  """
  Cluster anchors.
  """
  # initialize random centroids
  centroid_indices = np.random.choice(len(gt_boxes), n_anchors)
  centroids = []
  for centroid_index in centroid_indices:
    centroids.append(gt_boxes[centroid_index])

  # iterate k-means
  anchors, avg_iou, loss = run_k_mean(n_anchors, gt_boxes, centroids)

  while True:
    anchors, avg_iou, curr_loss = run_k_mean(n_anchors, gt_boxes, anchors)
    if abs(loss - curr_loss) < loss_convergence:
      break
    loss = curr_loss

  return anchors, avg_iou


def run_k_mean(n_anchors, boxes, centroids):
  '''
  Perform K-mean clustering on training ground truth to generate anchors.
  In the paper, authors argues that generating anchors through anchors would improve Recall of the network

  NOTE: Euclidean distance produces larger errors for larger boxes. Therefore, YOLOv2 did not use Euclidean distance
        to measure calculate loss. Instead, it uses the following formula:
        d(box, centroid)= 1 - IoU (box, centroid)

  :param n_anchors:
  :param boxes:
  :param centroids:
  :return:
      new_centroids: set of new anchors
      groups:        wth?
      loss:          compared to current bboxes
  '''
  loss = 0
  groups = []
  new_centroids = []
  for i in range(n_anchors):
    groups.append([])
    new_centroids.append(Box(0, 0, 0, 0))

  for box in boxes:
    min_distance = 1
    group_index = 0

    for i, centroid in enumerate(centroids):
      distance = 1 - box_iou(box, centroid)  # Used in YOLO9000
      if distance < min_distance:
        min_distance = distance
        group_index = i

    groups[group_index].append(box)
    loss += min_distance
    new_centroids[group_index].w += box.w
    new_centroids[group_index].h += box.h

  for i in range(n_anchors):
    if len(groups[i]) == 0:
      continue
    new_centroids[i].w /= len(groups[i])
    new_centroids[i].h /= len(groups[i])

  iou = 0
  counter = 0
  for i, anchor in enumerate(new_centroids):
    for gt_box in groups[i]:
      iou += box_iou(gt_box, anchor)
      counter += 1

  avg_iou = iou / counter
  return new_centroids, avg_iou, loss


if __name__ == '__main__':
  main()
