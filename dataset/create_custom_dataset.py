"""
This script creates a custom dataset for training and evaluation. It will generate:

   + a CSV training file
   + a categories text file to map indices to label names.
   + an anchor text file (depending on number of anchors, default = 5)

Requirement:
-----------
   + a text file, containing ground truths, in following format:
        image1.jpg , x1, y1, x2, y2, label
        image2.jpg, x1, y1, x2, y2, label2

    whereas:
       image1.jpg    : a str  - absolute image path
       x1, y1, x2, y2: float  - absolute object coordinates
       label:          string - name of label

Example usage:
-------------
python create_custom_dataset.py
 --data_file     /path/to/text_file.txt
 --num_anchors   5
 --output_dir   ./dataset/new_dataset
s
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
import csv
import numpy as np
from PIL import Image
from yolov2.utils.box import box_iou, Box
from config import IMG_INPUT_SIZE, SHRINK_FACTOR
from sklearn.model_selection import train_test_split

from argparse import ArgumentParser

parser = ArgumentParser(description="Generate custom dataset for training")

parser.add_argument('-p', '--path',
                    help="Path to text file", type=str, default=None)

parser.add_argument('-o', '--output_dir',
                    help="Path to output directory", type=str, default=None)

parser.add_argument('-n', '--number_anchors',
                    help="Number of anchors [default = 5]", type=int, default=5)


# @TODO: Stratifield Kfold for splitting dataset
def main():
    arguments = parser.parse_args()
    path = arguments.path
    output_dir = arguments.output_dir
    number_anchors = arguments.number_anchors
    split = arguments.split

    # #################################
    # Generate Anchors and Categories #
    # #################################
    with open(path, 'r') as f:
        gt_boxes = []
        categories = {}

        lines = f.readlines()
        print("Calculating Anchors using K-mean Clustering....")
        id = 0
        for line in lines:
            img_path, x1, y1, x2, y2, label = line.rstrip().split(",")
            xc, yc, w, h = convert_edges_to_centroid(x1, y1, x2, y2)

            with Image.open(img_path) as img:
                img_width, img_height = img.size

            aspect_ratio = [IMG_INPUT_SIZE / float(img_width), IMG_INPUT_SIZE / float(img_height)]
            box = Box(0, 0, float(w) * aspect_ratio[0] / SHRINK_FACTOR, float(h) * aspect_ratio[1] / SHRINK_FACTOR)
            gt_boxes.append(box)

            if label not in categories:
                categories[label] = id
                id += 1

        anchors, avg_iou = k_mean_cluster(number_anchors, gt_boxes)
        print("Number of anchors: {:2} | Average IoU:{:-4f}\n\n ".format(number_anchors, avg_iou))

    categories_file = os.path.join(output_dir, 'categories.txt')
    anchors_file = os.path.join(output_dir, 'anchors.txt')

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(categories_file, 'w') as f:
        for item in categories.items():
            f.write(item[0] + '\n')

    with open(anchors_file, 'w') as f:
        for anchor in anchors:
            f.write("({:5f}, {:5f})\n".format(anchor.w, anchor.h))

    # ###################################
    # Generate Training/Validation data #
    # ###################################
    print('Generating data..')
    if split is True:
        with open(path) as txt_file:
            x = txt_file.read().splitlines()[1:]
            x_train, x_test = train_test_split(x, test_size=0.2)

        training_path = os.path.join(output_dir, 'training_data.csv')
        testing_path = os.path.join(output_dir, 'validation_data.csv')

        save_dataset(x_train, training_path)
        save_dataset(x_test, testing_path)
    else:
        with open(path) as txt_file:
            data = txt_file.read().splitlines()
            training_path = os.path.join(output_dir, 'training_data.csv')
            save_dataset(data, training_path)
    print('Done')


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
