import numpy as np

IMG_INPUT     = 608  # DarkNet Feature Extractor uses (608, 608) image size
N_CLASSES     = 31
N_ANCHORS     = 5
SHRINK_FACTOR = 32    # **For DarkNet19 only (max-pool 5 times) ** 2^5 = 32
AUGMENT_LEVEL = 10    # The higher, the more data is augmented
MULTI_SCALE   = [1.0]
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
