import numpy as np

IMG_INPUT     = 960  # DarkNet Feature Extractor uses (608, 608) image size
N_CLASSES     = 31
N_ANCHORS     = 5
SHRINK_FACTOR = 32    # **For DarkNet19 only (max-pool 5 times) ** 2^5 = 32
AUGMENT_LEVEL = 15    # The higher, the more data is augmented
MULTI_SCALE   = [1.0]
ANCHORS       = np.array(((0.03283318210930825, 0.0483890193566751),
                          (0.059577141608391594, 0.08738709207459215),
                          (0.04450034340659346, 0.064308608058608),
                          (0.023717899133663362, 0.035715759075907606),
                          (0.08816276658767774, 0.1294924960505529)))
CATEGORIES   = np.array(('addedLane', 'bicyclesMayUseFullLane', 'curveLeft', 'curveRight', 'doNotEnter', 'intersection',
                         'intersectionLaneControl', 'keepRight', 'laneEnds',
                         'leftAndUTurnControl', 'merge', 'noLeftAndUTurn', 'noParking', 'noRightTurn',
                         'noUTurn', 'pedestrianCrossing', 'school', 'signalAhead',
                         'speedBumpsAhead', 'speedLimit15', 'speedLimit25', 'speedLimit30', 'speedLimit35', 'speedLimit40',
                         'speedLimit45', 'speedLimit50', 'speedLimit60', 'stop', 'stopAhead', 'yieldAhead', 'yieldToPedestrian'))
