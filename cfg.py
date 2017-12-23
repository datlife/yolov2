"""
Main configuration file for YOLOv2 detection model
"""
# Configuration for COCO dataset
IMG_INPUT_SIZE = 608
N_CLASSES = 80
N_ANCHORS = 5

# Map indices to actual label names - absolute path required
CATEGORIES = "./dataset/coco/categories.txt"
ANCHORS    = "./dataset/coco/anchors.txt"


# Type of Feature Extractor.
FEATURE_EXTRACTOR     = 'yolov2'

# Image resolution being reduced 2^5 = 32 times
SHRINK_FACTOR  = 32

# OPTIONAL -  HIERARCHICAL TREE - Disabled if training on your custom data (or look at lisa.tree on how to setup)
HIERARCHICAL_TREE_PATH   = "/home/dat/Documents/yolov2/dataset/combined_lisa/lisa.tree"
