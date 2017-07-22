import numpy as np

N_CLASSES     = 31
N_ANCHORS     = 5
SHRINK_FACTOR = 32   # **For DarkNet19 only (max-pool 5 times) ** 2^5 = 32
AUGMENT_LEVEL = 5    # The higher, the more data is augmented
MULTI_SCALE   = [1.0]
ANCHORS       = np.array(((3.5265106635071097, 3.884774881516587),
                         (0.9487159653465345, 1.0714727722772281),
                         (2.3830856643356637, 2.6216127622377647),
                         (1.7800137362637383, 1.92925824175824),
                         (1.31332728437233, 1.451670580700253)))

CATEGORIES   = np.array(('addedLane', 'bicyclesMayUseFullLane', 'curveLeft', 'curveRight', 'doNotEnter', 'intersection',
                         'intersectionLaneControl', 'keepRight', 'laneEnds',
                         'leftAndUTurnControl', 'merge', 'noLeftAndUTurn', 'noParking', 'noRightTurn',
                         'noUTurn', 'pedestrianCrossing', 'school', 'signalAhead',
                         'speedBumpsAhead', 'speedLimit15', 'speedLimit25', 'speedLimit30', 'speedLimit35', 'speedLimit40',
                         'speedLimit45', 'speedLimit50', 'speedLimit60', 'stop', 'stopAhead', 'yieldAhead', 'yieldToPedestrian'))
