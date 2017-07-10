"""
Generate anchors from training set using K-mean clustering

Work cited:
The K-mean function has been adopted from YOLOv2 Chainer Project
"""
import glob
import numpy as np
from utils.box import Box, box_iou

# @TODO: double-check box_iou


def k_mean_cluster(bboxes, num_anchors, centroids, loss_convergence=1e-5):
    """
    
    :param bboxes: 
    :param num_anchors: 
    :param centroids: 
    :param loss_convergence: 
    :return: 
    """
    anchors, groups, loss = run_k_mean(bboxes, num_anchors, centroids)
    while True:
        anchors, groups, curr_loss = run_k_mean(bboxes, num_anchors, anchors)
        # print("loss = %f" % curr_loss)
        if abs(loss - curr_loss) < loss_convergence:
            break
        loss = curr_loss

    return anchors


def run_k_mean(bboxes, num_anchors, centroids):
    """
    Perform K-mean clustering on training ground truth to generate anchors. 
    In the paper, authors argues that generating anchors through anchors would improve Recall of the network
    
    NOTE: Euclidean distance produces larger errors for larger boxes. Therefore, YOLOv2 did not use Euclidean distance 
          to measure calculate loss. Instead, it is used the following formula:
          
                    d(box, centroid) = 1−IOU(box, centroid)
    
    :param bboxes:      list of bounding box in format [x1, y1, w, h]
    :param num_anchors: K-value , number of desired anchors box
    :param centroids: 
    :return: 
        new_centroids: set of new anchors
        groups:        wth?
        loss:          compared to current bboxes
    """
    loss = 0
    groups = []
    new_centroids = []
    for i in range(num_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in bboxes:
        min_distance = 1
        group_index = 0

        for i, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))  # <--- Distance Error cost suggested by YOLO9000 paper
            if distance < min_distance:
                min_distance = distance
                group_index  = i

        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(num_anchors):
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss

if __name__ == "__main__":
    # hyper parameters
    label_path = "/home/ubuntu/dataset/training/"
    n_anchors  = 5
    image_width  = 1280
    image_height = 960
    grid_width   = 40
    grid_height  = 30

    boxes = []
    label_files = glob.glob("*.csv" % label_path)
    for label_file in label_files:
        with open(label_file, "r") as f:
            label, x, y, w, h = f.read().strip().split(" ")
            boxes.append(Box(0, 0, float(w), float(h)))

    # initial centroids
    centroid_indices = np.random.choice(len(boxes), n_anchors)
    anchors = []
    for centroid_index in centroid_indices:
        anchors.append(boxes[centroid_index])

    # iterate k-means
    anchors = k_mean_cluster(boxes, n_anchors, anchors, loss_convergence=1e-5)

    # print result
    for anchor in anchors:
        print(anchor.w * grid_width, anchor.h * grid_height)
