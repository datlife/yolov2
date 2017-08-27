from keras.applications.resnet50 import decode_predictions
from keras.preprocessing import image
from mobilenet import mobile_net, preprocess_input
import numpy as np

img_path = '../../test_imgs/dachshund.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

mobilenet = mobile_net()
mobilenet.load_weights('../../weights/mobilenet_1_0_224_tf.h5')
preds = mobilenet.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
