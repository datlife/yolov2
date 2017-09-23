'''
Main configuration file for YOLOv2 Project.
Modify every time one would like to train on a new dataset
'''
# Image input resolution. Higher resolution might improve accuracy but reduce the interference
IMG_INPUT_SIZE = 448

N_CLASSES = 20
N_ANCHORS = 5

# Type of Feature Extractor.   Currently supported:
#   'yolov2':     Original YOLOv2 feature extractor
#   'mobilenet' : MobileNet implementation from Google
#   'densenet'  : Densely Connected convolutional network (Facebook)
FEATURE_EXTRACTOR     = 'yolov2'

# Map indices to actual label names - absolute path required
CATEGORIES = "/home/dat/Documents/yolov2/dataset/pascal/categories.txt"
ANCHORS    = "/home/dat/Documents/yolov2/dataset/pascal/anchors.txt"

# If a feature extractor performed 5 max-pooling --> Image resolution being reduced 2^5 = 32 times
# Most current state-of-the-art models have max-pooling layers (August, 2017)
SHRINK_FACTOR  = 32


# OPTIONAL -  HIERARCHICAL TREE - Disabled if training on your custom data (or look at lisa.tree on how to setup)
HIERARCHICAL_TREE_PATH   = "/home/dat/Documents/yolov2/dataset/combined_lisa/lisa.tree"
