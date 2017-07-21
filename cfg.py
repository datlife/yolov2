import numpy as np

N_CLASSES     = 31
N_ANCHORS     = 5
SHRINK_FACTOR = 32   # **For DarkNet19 only (max-pool 5 times) ** 2^5 = 32
AUGMENT_LEVEL = 5    # The higher, the more data is augmented
MULTI_SCALE   = [1.0]
ANCHORS       = np.array(((3.526510663507109, 3.884774881516588),
                          (1.7800137362637363, 1.9292582417582418),
                          (0.9487159653465347, 1.0714727722772277),
                          (2.383085664335664, 2.621612762237762),
                          (1.3133272843723314, 1.4516705807002561)))
CATEGORIES   = np.array(('addedLane', 'bicyclesMayUseFullLane', 'curveLeft', 'curveRight', 'doNotEnter', 'intersection',
                         'intersectionLaneControl', 'keepRight', 'laneEnds',
                         'leftAndUTurnControl', 'merge', 'noLeftAndUTurn', 'noParking', 'noRightTurn',
                         'noUTurn', 'pedestrianCrossing', 'school', 'signalAhead',
                         'speedBumpsAhead', 'speedLimit15', 'speedLimit25', 'speedLimit30', 'speedLimit35', 'speedLimit40',
                         'speedLimit45', 'speedLimit50', 'speedLimit60', 'stop', 'stopAhead', 'yieldAhead', 'yieldToPedestrian'))
