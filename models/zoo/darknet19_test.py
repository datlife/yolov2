import cv2
import numpy as np
from darknet19 import darknet19, yolo_preprocess_input
from decode_imagenet import decode_predictions

# Set up model
darknet = darknet19(input_size=(448, 448, 3), include_top=True)
darknet.load_weights('../../weights/feature_extractors/darknet19_448.h5')
darknet.summary()


def predict_one_example():
    # Prepare data input
    img_path = '../../test_imgs/cat.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (448, 448))
    x = yolo_preprocess_input(img)
    x = np.expand_dims(x, axis=0)
    preds = darknet.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])


# Measure performance
import timeit

performance = timeit.repeat('predict_one_example()', "from __main__ import predict_one_example", number=1, repeat=50)
avg = np.average(performance[1:])
print ("Average forward pass : {} ms".format(avg * 1000))
