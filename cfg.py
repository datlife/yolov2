IMG_INPUT = 960
N_CLASSES = 61
N_ANCHORS  = 5
SHRINK_FACTOR = 32  # Max-pooling 5 times --> 2^5 = 32 (feature map size is reduced 32 times)

CATEGORIES = "/home/dat/Documents/yolov2/dataset/combined_lisa/categories_tree.txt"

# Combined LISA Dataset
ANCHORS = [(1.2255253619, 2.0790172001), (0.88504653628, 1.46357817461), (0.629442321737, 1.07695834955),
           (1.85679560768, 3.06510868443), (0.426883396719, 0.751765360431)]

# OPTIONAL -  HIERARCHICAL TREE - Please disabled if training on your custom data (or look at lisa.tree on how to setup)
ENABLE_TREE = True
TREE_FILE = "/home/dat/Documents/yolov2/dataset/combined_lisa/lisa.tree"


# COCO Dataset
# ANCHORS    = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)]

# LISA Extension Dataset
# ANCHORS = [(0.845506524725, 1.22186355311), (0.45064008354, 0.678599422442), (1.67509256517, 2.46035742496),
#            (0.623830460077, 0.919391367777), (1.13196569056, 1.66035474942)]
