'''
Main configuration file for YOLOv2 Project.

Modify every time one would like to train on a new dataset
'''

# Type of Feature Extractor.   Currently supported:
#   'yolov2':     Original YOLOv2 feature extractor
#   'mobilenet' : MobileNet implementation from Google
#   'densenet'  : Densely Connected convolutional network (Facebook)
FEATURE_EXTRACTOR     = 'yolov2'

# Map indices to actual label names - absolute path required
CATEGORIES = "/home/dat/Documents/yolov2/dataset/coco/categories.txt"
ANCHORS    = "/home/dat/Documents/yolov2/dataset/coco/anchors.txt"

# OPTIONAL -  HIERARCHICAL TREE - Please disabled if training on your custom data (or look at lisa.tree on how to setup)
ENABLE_HIERARCHICAL_TREE = False
HIERARCHICAL_TREE_PATH   = "/home/dat/Documents/yolov2/dataset/combined_lisa/lisa.tree"

# Image input resolution. Higher resolution might improve accuracy but reduce the interference
IMG_INPUT_SIZE = 608

# Number of classes in the data set (included the abstracted objects if hierarchical tree is enabled)
N_CLASSES      = 80

# Number of anchors
N_ANCHORS      = 5

# If a feature extractor performed 5 max-pooling --> Image resolution being reduced 2^5 = 32 times
# Most current state-of-the-art models have max-pooling layers (August, 2017)
SHRINK_FACTOR  = 32
