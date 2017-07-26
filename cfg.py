import numpy as np

N_ANCHORS     = 5      # Number of ANCHORS (in YOLO9000 paper, 5 was chosen for optimal speed/performance)
N_CLASSES     = 56     # Number of classes
CATEGORIES    = 'lisa.names'

# Optional
IMG_INPUT     = 672    # DarkNet Feature Extractor uses (608, 608) image size
SHRINK_FACTOR = 32     # How much image dimension has been reduced. **For DarkNet19 (max-pool 5 times) ** 2^5 = 32
AUGMENT_LEVEL = 15     # The higher, the more data is augmented

MULTI_SCALE   = [1.0]  # For Multi-scale training proposed in YOLO9000 - not working yet

# Relative anchors (in percentage) to the output feature map
ANCHORS       = np.array(((1.67509256517, 2.46035742496),
                          (1.13196569056, 1.66035474942),
                          (0.623830460077, 0.919391367777),
                          (0.45064008354, 0.678599422442),
                          (0.845506524725, 1.22186355311)))
print(ANCHORS)
anchors = ANCHORS / np.array([608/32., 608/32.])
ANCHORS = anchors * np.array([672./32, 672./32])
print ANCHORS

