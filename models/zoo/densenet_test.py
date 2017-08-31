import cv2
import timeit
import numpy as np
from decode_imagenet import decode_predictions
from darknet19 import yolo_preprocess_input
from densenet import densenet
from keras.models import load_model, Model
from keras.layers import Activation

# Set up model
densenet = densenet(include_top=True)


def predict_one_example():
    # Prepare data input
    img_path = '../../test_imgs/cat.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (608, 608))
    x = yolo_preprocess_input(img)
    x = np.expand_dims(x, axis=0)
    preds = densenet.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])


# Measure performance
performance = timeit.repeat('predict_one_example()', "from __main__ import predict_one_example", number=1, repeat=50)
avg = np.average(performance[1:])
print ("Average forward pass : {} ms".format(avg * 1000))
# Average forward pass : 51.1690645802 ms
