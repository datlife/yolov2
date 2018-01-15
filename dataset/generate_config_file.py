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
import os
import csv
import time
import random
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
  start_reading_time = time.time()
  with open(data_file, 'r') as fio:
    print("[INFO] Reading data")
    gt_boxes = []
    next(fio)  # skip header line @TODO: handle this in better way
    for line in fio:
      img_path, x1, y1, x2, y2, label = line.rstrip().split(",")
      xc, yc, w, h = convert_edges_to_centroid(x1, y1, x2, y2)
      with Image.open(img_path) as img:
        img_width, img_height = img.size

      # Convert to relative bboxes
      relative_bbox = np.array([0, 0, w, h]) / np.array([img_width, img_height,
                                                         img_width, img_height],
                                                        dtype=np.float)
      gt_boxes.append(relative_bbox)

    completed_reading_time = time.time()
    print("[INFO] File is loaded in {:3} secs".format(completed_reading_time - start_reading_time))
    print("[INFO] Running K mean clustering")
    anchors, avg_iou = k_mean_cluster(num_anchors, np.array(gt_boxes))
    print("[INFO] Completed in {:2f} secs".format(time.time() - completed_reading_time))
    # print("Number of anchors: {:2} | Average IoU:{:-4f}\n ".format(num_anchors, avg_iou))
    print(avg_iou)
    print(anchors[..., 2:4])
  # ###################################
  # Generate Training/Validation data #
  # ###################################


def k_mean_cluster(num_anchors, gt_boxes, loss_convergence=1e-7):
  """Cluster anchors.
  """
  # Randomly pick 5 boxes as centroids
  centroids = np.array(random.sample(gt_boxes, k=num_anchors))
  anchors, loss = run_k_mean(num_anchors, gt_boxes, centroids)
  while True:
    anchors, curr_loss = run_k_mean(num_anchors, gt_boxes, anchors)
    if abs(loss - curr_loss) < loss_convergence:
      break
    loss = curr_loss

  ious    = compute_iou(gt_boxes, anchors)
  avg_iou = np.mean(np.max(ious, axis=-1))
  return anchors, avg_iou


def run_k_mean(num_anchors, boxes, centroids):
  """Perform K-mean clustering on training ground truth to generate anchors.

  In the paper, authors argues that generating anchors through k-mean clustering would improve Recall of the network

  NOTE:YOLOv2 uses the following formula to calculate distance:
            d(box, centroid)= 1 - IoU (box, centroid)
  Args:
    num_anchors:
    boxes:
    centroids:

  Returns:
      new_centroids: set of new anchors
      groups:        wth?
      loss:          compared to current bboxes
  """
  # Initialize updated anchors with 0.0
  updated_centroids = np.zeros_like(centroids)

  # Measure distances between all bboxes and current centroids
  distances = 1.0 - compute_iou(boxes, centroids)

  # Find the closest centroids which a box should belong too
  closest_centroids = np.argmin(distances, axis=-1)

  # Now, update centroids by taking average of xc, yc, w, h
  for box, idx in zip(boxes, closest_centroids):
    updated_centroids[idx] += box

  for i, freq in zip(*(np.unique(closest_centroids, return_counts=True))):
    updated_centroids[i] /= freq

  loss    = np.sum(np.min(distances, axis=-1))
  return updated_centroids, loss


def compute_iou(boxes1, boxes2):
  """
  Args:
    boxes1: - a np.array of N boxes [N, 4]
    boxes2: - a np.array of M boxes [M, 4]

  Returns:
    a IoU matrix [N, M]

  """
  areas1  = area(boxes1)
  areas2  = area(boxes2)
  intersections = intersection(boxes1, boxes2)

  unions = (np.expand_dims(areas1, 1) +
            np.expand_dims(areas2, 0) - intersections)

  return np.where(np.equal(intersections, 0.0),
                  np.zeros_like(intersections),
                  np.divide(intersections, unions))


def area(boxes):
  """Computes area of boxes.

  Args:
    boxes: a np.array holding N boxes [N, 4]

  Returns:
    a tensor with shape [N] representing box areas.
  """
  y_min, x_min, y_max, x_max = np.split(boxes, 4, axis=-1)
  return np.squeeze((y_max - y_min) * (x_max - x_min), [1])


def intersection(boxes1, boxes2):
  """Compute pairwise intersection areas between boxes.
  Args:
    boxes1:   a np.array holding N boxes [N, 4]
    boxes2:  a np.array holding M boxes [N, 4]

  Returns:
      a np.array shape [N, M] representing pairwise intersections
  """
  y_min1, x_min1, y_max1, x_max1 = np.split(boxes1, 4, axis=-1)
  y_min2, x_min2, y_max2, x_max2 = np.split(boxes2, 4, axis=-1)

  all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
  all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))

  intersect_heights  = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
  all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
  all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))

  intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)

  return intersect_heights * intersect_widths


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


if __name__ == '__main__':
  main()

 # for box in boxes:
 #    min_distance = 1
 #    group_index = 0
 #
 #    for i, centroid in enumerate(centroids):
 #      # Used in YOLO9000
 #      distance = 1 - box_iou(box, centroid)
 #      if distance < min_distance:
 #        min_distance = distance
 #        group_index = i
 #
 #    groups[group_index].append(box)
 #    loss += min_distance
 #    new_centroids[group_index].w += box.w
 #    new_centroids[group_index].h += box.h
 #
 #  for i in range(num_anchors):
 #    if len(groups[i]) == 0:
 #      continue
 #    new_centroids[i].w /= len(groups[i])
 #    new_centroids[i].h /= len(groups[i])
 #
 #  iou = 0
 #  counter = 0
 #  for i, anchor in enumerate(new_centroids):
 #    for gt_box in groups[i]:
 #      iou += box_iou(gt_box, anchor)
 #      counter += 1
