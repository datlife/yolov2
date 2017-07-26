import numpy as np

N_ANCHORS     = 5      # Number of ANCHORS (in YOLO9000 paper, 5 was chosen for optimal speed/performance)
N_CLASSES     = 31     # Number of classes
IMG_INPUT     = 608    # DarkNet Feature Extractor uses (608, 608) image size

SHRINK_FACTOR = 32     # How much image dimension has been reduced. **For DarkNet19 (max-pool 5 times) ** 2^5 = 32
AUGMENT_LEVEL = 10     # The higher, the more data is augmented

MULTI_SCALE   = [1.0]  # For Multi-scale training proposed in YOLO9000 - not working yet
ANCHORS       = np.array(((1.67509256517, 2.46035742496),
                          (1.13196569056, 1.66035474942),
                          (0.623830460077, 0.919391367777),
                          (0.45064008354, 0.678599422442),
                          (0.845506524725, 1.22186355311)))


CATEGORIES   = np.array(('addedLane', 'bicyclesMayUseFullLane', 'curveLeft', 'curveRight', 'doNotEnter', 'intersection',
                         'intersectionLaneControl', 'keepRight', 'laneEnds',
                         'leftAndUTurnControl', 'merge', 'noLeftAndUTurn', 'noParking', 'noRightTurn',
                         'noUTurn', 'pedestrianCrossing', 'school', 'signalAhead',
                         'speedBumpsAhead', 'speedLimit15', 'speedLimit25', 'speedLimit30', 'speedLimit35', 'speedLimit40',
                         'speedLimit45', 'speedLimit50', 'speedLimit60', 'stop', 'stopAhead', 'yieldAhead', 'yieldToPedestrian'))
