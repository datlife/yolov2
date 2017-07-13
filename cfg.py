import numpy as np

IMG_WIDTH     = 1280
IMG_HEIGHT    = 960
SHRINK_FACTOR = 32   # Darknet19 uses max-pool 5 times, 2^5 = 32
GRID_W        = IMG_WIDTH/SHRINK_FACTOR
GRID_H        = IMG_HEIGHT/SHRINK_FACTOR
N_CLASSES     = 31
ANCHORS       = np.array(((0.57273, 0.677385),
                         (1.87446, 2.06253),
                         (3.33843, 5.47434),
                         (7.88282, 3.52778),
                         (9.77052, 9.16828)))
