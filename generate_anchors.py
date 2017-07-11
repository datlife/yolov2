import numpy as np
from utils.box import Box, box_iou
from argparse import ArgumentParser

parser = ArgumentParser(description="Generate Anchors from ground truth boxes using K-mean clustering")

parser.add_argument('--num_anchors', help="Number of anchors",      type=int,  default=5)
parser.add_argument('--label_path',  help="Text file [label, x1, y1, w, h]", default='training.txt')
parser.add_argument('--loss',        help="Loss Convergence value", type=float,  default=1e-5)
parser.add_argument('--img_width',   help='Image width',  type=int, default=1280)
parser.add_argument('--img_height',  help='Image height', type=int, default=960)


def __main__():
    args = parser.parse_args()
    # Extract Arguments
    k            = args.num_anchors
    label_path   = args.label_path
    loss_conv    = args.loss
    img_width    = args.img_width
    img_height   = args.img_height
    img_size     = (img_width, img_height, 3)
    gt_boxes     = []
    feature_size = 1/32   # since DarkNet performs max-pool 5 times -> feature map size shrinks 2^5 =32 times

    # Extract bounding boxes from training data
    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            img_path, x1, y1, x2, y2, label = line.rstrip().split(",")
            xc, yc, w, h = convert_bbox(x1, y1, x2, y2)
            xc, yc, w, h = scale_rel_box(img_size, Box(xc, yc, w, h))
            # since we calculate w h of anchors, we do not take xc yc into account
            gt_boxes.append(Box(0, 0, float(w), float(h)))
    print("Number of ground truth boxes: {} boxes".format(len(gt_boxes)))
   
    # K-MEAN CLUSTERING
    anchors, avg_iou = k_mean_cluster(k, gt_boxes, loss_convergence=loss_conv)
    print("K = : {:2} | AVG_IOU:{:-4f} ".format(k, avg_iou))

    # print result
    print("Anchors box result [relative to feature map]:\n")
    for anchor in anchors:
        print("({}, {})".format(anchor.w * img_width*feature_size, anchor.h * img_height*feature_size))


def k_mean_cluster(k, gt_boxes, loss_convergence=1e-5):
    """
    Cluster anchors.
    """
    # initial random centroids
    centroid_indices = np.random.choice(len(gt_boxes), k)
    centroids = []
    for centroid_index in centroid_indices:
        centroids.append(gt_boxes[centroid_index])

    # iterate k-means
    anchors, avg_iou, loss = run_k_mean(k, gt_boxes, centroids)
    while True:
        anchors, avg_iou, curr_loss = run_k_mean(k, gt_boxes, anchors)
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

    :param num_anchors: K-value , number of desired anchors box
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
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = i

        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        if (len(groups[i]) == 0):
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


def scale_rel_box(img_size, box):
    """
    Scale bounding box relative to image size
    """
    width, height, _ = img_size
    dw = 1. / width
    dh = 1. / height
    xc = box.x * dw
    yc = box.y * dh
    w  = box.w * dw
    h  = box.h * dh
    return xc, yc, w, h


if __name__ == "__main__":
    __main__()
