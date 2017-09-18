import cv2
import numpy as np
from keras.applications.resnet50 import decode_predictions

from mobilenet import mobile_net, preprocess_input

# Set up model
mobilenet = mobile_net(input_size=(224, 224, 3), include_top=True)
mobilenet.load_weights('../../weights/feature_extractors/mobilenet_1_0_224_tf.h5')
mobilenet.summary()


def predict_one_example():
    # Prepare data input
    img_path = '../../test_imgs/cat.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    x = preprocess_input(img)
    x = np.expand_dims(x, axis=0)
    preds = mobilenet.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])


# Measure performance
import timeit

performance = timeit.repeat('predict_one_example()', "from __main__ import predict_one_example", number=1, repeat=50)
avg = np.average(performance[1:])
print ("Average forward pass : {} ms".format(avg * 1000))
# Average forward pass : 16.3480602965 ms
