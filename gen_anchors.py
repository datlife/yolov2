# coding: utf-8

"""
This script will generate anchors based on your training bounding boxes.
It will apply k-mean cluster through the boxes in training data to determine size (w, h) for K anchors (in YOLOv2, K = 5)

Requirements
------------
   1. One needs to have a training data in txt format as [img_path, x1, x2, y1, y2, label]
   2. Image size of training data. [default : w= 1280 h =960]


Example:
--------

python gen_anchors.py --num_anchors 5 --label_bath training.txt --img_width 1280 --img_height 960
python gen_anchors.py -n 5 -p training.txt -w 1280 -h 960
s
"""
import cv2
from utils.box import Box, box_iou
from argparse import ArgumentParser
from cfg import *


parser = ArgumentParser(description="Generate Anchors from ground truth boxes using K-mean clustering")
parser.add_argument('-n', '--num_anchors',
                    type=int,   default=5,
                    help="Number of anchors")
parser.add_argument('-p',
                    '--label_path',
                    type=str, default='data/training.txt',
                    help="Path to Training txt file")
parser.add_argument('-width',
                    '--image_width',
                    type=int, default=1280,
                    help="Image Width")
parser.add_argument('-height',
                    '--image_height',
                    type=int, default=960,
                    help="Image Height")


def __main__():
    args = parser.parse_args()
    k            = args.num_anchors
    label_path   = args.label_path
    img_width    = args.image_width
    img_height   = args.image_height
    gt_boxes     = []

    # Extract bounding boxes from training data
    with open(label_path, "r") as f:
        lines = f.readlines()
        # Update aspect ratio
        aspect_ratio = [IMG_INPUT / float(img_width), IMG_INPUT / float(img_height)]
        print(aspect_ratio)
        for line in lines:
            img_path, x1, y1, x2, y2, label = line.rstrip().split(",")
            xc, yc, w, h = convert_bbox(x1, y1, x2, y2)
            box = Box(0, 0, float(w) * aspect_ratio[0] / SHRINK_FACTOR, float(h) * aspect_ratio[1] / SHRINK_FACTOR)
            gt_boxes.append(box)

    # ############## K-MEAN CLUSTERING ########################
    anchors, avg_iou = k_mean_cluster(k, gt_boxes)
    print("K = : {:2} | AVG_IOU:{:-4f} ".format(k, avg_iou))

    # print result
    for anchor in anchors:
        print("({}, {})".format(anchor.w, anchor.h))


def k_mean_cluster(n_anchors, gt_boxes, loss_convergence=1e-5):
    """
    Cluster anchors.
    """
    # initial random centroids
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
    """
    Perform K-mean clustering on training ground truth to generate anchors. 
    In the paper, authors argues that generating anchors through anchors would improve Recall of the network

    NOTE: Euclidean distance produces larger errors for larger boxes. Therefore, YOLOv2 did not use Euclidean distance 
          to measure calculate loss. Instead, it uses the following formula:

                    d(box, centroid) = 1−IOU(box, centroid)

    :param n_anchors: K-value , number of desired anchors box
    :param boxes:      list of bounding box in format [x1, y1, w, h]
    :param centroids: 
    :return: 
        new_centroids: set of new anchors
        groups:        wth?
        loss:          compared to current bboxes
    """
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
            distance = 1 - box_iou(box, centroid) # Used in YOLO9000
            if (distance < min_distance):
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
    # print("Average IOU: {:4f}".format(avg_iou))
    return new_centroids, avg_iou, loss


def convert_bbox(x1, y1, x2, y2):
    w = float(x2) - float(x1)
    h = float(y2) - float(y1)
    xc = float(x1) + w / 2.
    yc = float(y1) + h / 2.
    return xc, yc, w, h


class Box(object):
    def __init__(self, xc, yc, w, h):
        self.x = xc
        self.y = yc
        self.w = w
        self.h = h


def box_iou(b1, b2):
    intersect = box_intersection(b1, b2)
    union = box_union(b1, b2)
    iou = float(intersect / union)
    return iou


def box_intersection(b1, b2):
    w = overlap(b1.x, b1.w, b2.x, b2.w)
    h = overlap(b1.x, b1.h, b2.x, b2.h)
    if (w < 0) or (h < 0): return 0
    area = w * h
    return area


def overlap(x1, w1, x2, w2):
    l1 = x1 - (w1 / 2.)
    l2 = x2 - (w2 / 2.)
    r1 = x1 + (w1 / 2.)
    r2 = x2 + (w2 / 2.)
    left = l1 if l1 >= l2 else l2
    right = r1 if r1 <= r2 else r2
    return right - left


def box_union(b1, b2):
    intersect = box_intersection(b1, b2)
    union = (b1.w * b1.h) + (b2.w * b2.h) - intersect
    return union


if __name__ == "__main__":
    __main__()
