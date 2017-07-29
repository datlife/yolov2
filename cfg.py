import numpy as np
from softmaxtree.Tree import SoftMaxTree

N_ANCHORS     = 5      # Number of ANCHORS (in YOLO9000 paper, 5 was chosen for optimal speed/performance)
N_CLASSES     = 61     # Number of classes
CATEGORIES    = 'lisa.categories'  # Order is matter, be careful of extra spaces - No space is allowed
HIER_TREE     = SoftMaxTree(tree_file='lisa.tree')  # Hierarchical Soft-max Implementation

# Optional
IMG_INPUT     = 608    # DarkNet Feature Extractor uses (608, 608) image size
SHRINK_FACTOR = 32     # How much image dimension has been reduced. **For DarkNet19 (max-pool 5 times) ** 2^5 = 32
AUGMENT_LEVEL = 10     # The higher, the more data is augmented
MULTI_SCALE   = [1.0]  # For Multi-scale training proposed in YOLO9000 - not working yet

# Relative anchors (in percentage) to the output feature map
# Imagine Anchor as pre-define width/height based on ground truth to help network converge faster
# ANCHORS       = np.array(((1.67509256517, 2.46035742496),
#                           (1.13196569056, 1.66035474942),
#                           (0.623830460077, 0.919391367777),
#                           (0.45064008354, 0.678599422442),
#                           (0.845506524725, 1.22186355311)))

ANCHORS       = np.array(((1.2255253619, 2.0790172001),
                          (1.85679560768, 3.06510868443),
                          (0.629442321737, 1.07695834955),
                          (0.426883396719, 0.751765360431),
                          (0.88504653628, 1.46357817461)))

# ANCHORS  *= np.array([672./608., 672./608.])
