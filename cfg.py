'''
Main configuration file for YOLOv2 Project


MODEL_TYPE: feature extractor definition
    Currently supported:
        'yolov2':     Original YOLOv2 feature extractor
        'mobilenet' : MobileNet implementation from Google
        'densenet'  : Densely Connected convolutional network (Facebook)

'''
MODEL_TYPE = 'yolov2'  # feature extractor definition
IMG_INPUT_SIZE = 480  # image resolution for inputs
N_CLASSES = 61
N_ANCHORS = 5
SHRINK_FACTOR = 32  # Max-pooling 5 times --> 2^5 = 32 (feature-map size is reduced 32 times)

CATEGORIES = "/home/dat/Documents/yolov2/dataset/combined_lisa/categories_tree.txt"

# Combined LISA Dataset
ANCHORS = [(1.2255253619, 2.0790172001), (0.88504653628, 1.46357817461), (0.629442321737, 1.07695834955),
           (1.85679560768, 3.06510868443), (0.426883396719, 0.751765360431)]

# OPTIONAL -  HIERARCHICAL TREE - Please disabled if training on your custom data (or look at lisa.tree on how to setup)
ENABLE_TREE = True
TREE_FILE = "/home/dat/Documents/yolov2/dataset/combined_lisa/lisa.tree"


# COCO Dataset
# ANCHORS    = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)]
