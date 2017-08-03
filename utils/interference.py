import PIL
from model.loss import custom_loss
from model.darknet19 import darknet19
from model.MobileYolo import MobileYolo
from cfg import *

WEIGHTS = './yolov2.weights'

class Interference():
    def __init__(self, model):
        """
        :param keras_model:
        """
        self.model = model
        self.model.compile(optimizer='adam', loss=custom_loss)





if __name__ == "__main__":
    # Build Model
    darknet    = darknet19(input_size=(IMG_INPUT, IMG_INPUT, 3))
    yolov2     = MobileYolo(feature_extractor=darknet,num_anchors=N_ANCHORS,
                            num_classes=N_CLASSES, fine_grain_layer=['leaky_re_lu_13', 'leaky_re_lu_8'])

    yolov2.model.load_weights(WEIGHTS)

    interference = Interference(yolov2.model)

    while True:
