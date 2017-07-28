import numpy as np
from softmaxtree.Tree import SoftMaxTree

N_ANCHORS     = 5      # Number of ANCHORS (in YOLO9000 paper, 5 was chosen for optimal speed/performance)
N_CLASSES     = 56     # Number of classes
CATEGORIES    = 'lisa.names'
HIER_TREE     = SoftMaxTree(tree_file='lisa.tree')  # Hierarchical Soft-max Implementation

# Optional
IMG_INPUT     = 608    # DarkNet Feature Extractor uses (608, 608) image size
SHRINK_FACTOR = 32     # How much image dimension has been reduced. **For DarkNet19 (max-pool 5 times) ** 2^5 = 32
AUGMENT_LEVEL = 2      # The higher, the more data is augmented

MULTI_SCALE   = [1.0]  # For Multi-scale training proposed in YOLO9000 - not working yet

# Relative anchors (in percentage) to the output feature map
ANCHORS       = np.array(((1.67509256517, 2.46035742496),
                          (1.13196569056, 1.66035474942),
                          (0.623830460077, 0.919391367777),
                          (0.45064008354, 0.678599422442),
                          (0.845506524725, 1.22186355311)))

from utils.parse_txt_to_inputs import parse_txt_to_inputs    # Data handler for LISA dataset

x_train, y_train = parse_txt_to_inputs('training_extension.txt')
labels           = np.unique(y_train[:,1])
print(labels)