'''Main configuration file for YOLOv2 Project
'''

# feature extractor definition.   Currently supported (take a look at models/zoo)
#   'yolov2':     Original YOLOv2 feature extractor
#   'mobilenet' : MobileNet implementation from Google
#   'densenet'  : Densely Connected convolutional network (Facebook)
FEATURE_EXTRACTOR     = 'yolov2'

# Image input resolution. Higher resolution might improve accuracy but might reduce the interference
IMG_INPUT_SIZE = 480

# Number of classes in the data set (included the abstracted objects if hierarchical tree is enabled)
N_CLASSES      = 61

# Number of anchors
N_ANCHORS      = 5

# Map indices to actual label names
CATEGORIES = "./dataset/combined_lisa/categories_tree.txt"
ANCHORS    = "./dataset/combined_lisa/anchors.txt"


# OPTIONAL -  HIERARCHICAL TREE - Please disabled if training on your custom data (or look at lisa.tree on how to setup)
ENABLE_TREE = True
TREE_FILE = "/home/dat/Documents/yolov2/dataset/combined_lisa/lisa.tree"

SHRINK_FACTOR  = 32  # Feature extractor performed 5 max-pooling --> Image resolution being reduced 2^5 = 32 times
