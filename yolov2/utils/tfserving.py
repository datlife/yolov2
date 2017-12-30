from __future__ import print_function
from __future__ import absolute_import

import os
import time
import signal
import subprocess

import numpy as np
import tensorflow as tf

# TensorFlow serving python API to send messages to server
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

DEFAULT_ACTIVATION_COMMAND = "tensorflow_model_server --port={} --model_name={} --model_base_path={}"


class DetectionClient(object):

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


class DetectionServer(object):
    def __init__(self, model, model_path, port=9000):
        self.server = None
        self.running = False
        self.model_path = model_path
        self.model = model
        self.port = port

    def is_running(self):
        return self.running

    def start(self):
        return self._callback(command='start')

    def stop(self):
        return self._callback(command='stop')

    def _callback(self, command):
        if command == 'start':
            if not self.running:
                print("Tensorflow Serving Server is launching ... ")
                self.server = subprocess.Popen(DEFAULT_ACTIVATION_COMMAND.format(
                    self.port,
                    self.model,
                    self.model_path),
                    stdin=subprocess.PIPE, shell=True)
                print("TF Serving Server is started at PID %s\n" % self.server.pid)
                self.running = True

            else:
                print("Tensorflow Serving Server has been activated already..\n")

        if command == 'stop':
            if self.running:
                self.running = True
                self._turn_off_server()

                print("Tensorflow Serving Server is off now\n")
            else:
                print("Tensorflow Serving Server is not activated yet..\n")

    def _turn_off_server(self):
        ps_command = subprocess.Popen("ps -o pid --ppid %d --noheaders" % self.server.pid,
                                      shell=True,
                                      stdout=subprocess.PIPE)

        ps_output = ps_command.stdout.read()
        return_code = ps_command.wait()
        for pid_str in ps_output.split("\n")[:-1]:
            os.kill(int(pid_str), signal.SIGINT)
        self.server.terminate()
