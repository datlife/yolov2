from __future__ import print_function
from __future__ import absolute_import
import time

import numpy as np
import tensorflow as tf

# TensorFlow serving python API to send messages to server
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


class ObjectDetectionClient(object):

    def __init__(self, server, model, label_dict, verbose=False):
        self.host, self.port = server.split(':')
        self.model      = model
        self.label_dict = label_dict
        self.verbose    = verbose

        channel   = implementations.insecure_channel(self.host, int(self.port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    def predict(self, image, img_dtype=tf.float32):
        request = predict_pb2.PredictRequest()

        start = time.time()
        image = np.expand_dims(image, axis=0)

        request.inputs['inputs'].CopyFrom(tf.make_tensor_proto(image,
                                                               dtype=img_dtype))
        request.model_spec.name           = self.model
        request.model_spec.signature_name = 'predict_images'

        pred  = time.time()
        result = self.stub.Predict(request, 20.0)  # 20 secs timeout

        if self.model == 'yolov2':
            num_detections = -1
        else:
            num_detections = int(result.outputs['num_detections'].float_val[0])

        classes = result.outputs['detection_classes'].float_val[:num_detections]
        scores  = result.outputs['detection_scores'].float_val[:num_detections]
        boxes   = result.outputs['detection_boxes'].float_val[:num_detections * 4]
        classes = [self.label_dict[int(idx)] for idx in classes]
        boxes   = [boxes[i:i + 4] for i in range(0, len(boxes), 4)]
        if self.verbose:
            print("Number of detections: %s" % len(classes))
            print("Server Prediction in {:.3f} sec || Total {:.3} sec".format(time.time() - pred, time.time() - start))

        return boxes, classes, scores


if __name__ == '__main__':
    import timeit
    setup = "from __main__ import ObjectDetectionClient, cv2"
    command = "ObjectDetectionClient('localhost:9000','yolov2', parse_config[ verbose=True).predict(cv2.imread('./test_imgs/person.jpg'))"
    print(timeit.timeit(command, setup=setup))