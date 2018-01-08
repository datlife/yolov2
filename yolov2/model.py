"""
Build YOLOv2 Model

The idea is that YOLOv2 consists of feature extractor and detector.
By using YOLOv2MetaArch, one can swap different types o feature extractor (DarkNet19, MobileNet, NASNet, DenseNet)
and different types of detector, too.

In this file, we construct a standard YOLOv2 using Darknet19 as feature extractor.
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model

from yolov2.utils.generator import TFData
from yolov2.utils.parser import parse_inputs, parse_label_map

from yolov2.core.loss import YOLOV2Loss
from yolov2.core.net_builder import YOLOv2MetaArch
from yolov2.core.custom_layers import Preprocessor, Reroute, OutputInterpreter

K = tf.keras.backend


# @TODO: add 10 samples images and draw bounding boxes + ground truths using IoU = 0.5, scores=0.7

class YOLOv2(object):

  def __init__(self, is_training, feature_extractor, detector, config_dict, add_summaries=True):

    self.config = config_dict
    self.is_training = is_training

    self.anchors = np.array(config_dict['anchors'])
    self.num_classes = config_dict['model']['num_classes']
    self.label_dict = parse_label_map(config_dict['label_map'])

    self.model = self._construct_model(is_training, feature_extractor, detector)
    self.summary = add_summaries

  def train(self, training_data, epochs, steps_per_epoch, batch_size, learning_rate, test_size=0.2):

    # ###############
    # Prepare Data  #
    # ###############
    inv_map = {v: k for k, v in self.label_dict.items()}
    filenames, labels = parse_inputs(training_data, inv_map)
    tfdata = TFData(self.num_classes, self.anchors, self.config['model']['shrink_factor'])

    # ###############
    # Compile model #
    # ###############
    loss = YOLOV2Loss(self.anchors, self.num_classes, summary=True)
    self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                       loss=loss.compute_loss)
    # ####################
    # Enable Tensorboard #
    # ####################
    summary_dir = self.config['training_params']['backup_dir']
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summary_dir + "/training", graph=K.get_session().graph)
    val_writer = tf.summary.FileWriter(summary_dir + "/validation")

    # ###############
    # Train model   #
    # ###############
    model = tf.keras.estimator.model_to_estimator(keras_model=self.model,
                                                  custom_objects={'Preprocessor': Preprocessor,
                                                                  'Reroute': Reroute,
                                                                  'OutputInterpreter': OutputInterpreter})

    for current_epoch in range(1, epochs + 1):
      # @TODO: Multi-scale training
      image_size = self.config['model']['image_size']
      x_train, x_val = train_test_split(filenames, test_size=test_size)
      y_train = [labels[k] for k in x_train]
      y_val = [labels[k] for k in x_val]

      print("Starting Training...")
      model.train(input_fn=lambda: tfdata.generator(x_val, y_val, image_size, batch_size),
                  steps=steps_per_epoch)


  def evaluate(self, testing_data, summaries=True):
    raise NotImplemented

  def _construct_model(self, is_training, feature_extractor, detector):

    yolov2 = YOLOv2MetaArch(feature_extractor=feature_extractor,
                            detector=detector,
                            anchors=self.anchors,
                            num_classes=self.num_classes)

    inputs = Input(shape=(None, None, 3), name='input_images')
    outputs = yolov2.predict(inputs)
    if is_training:
      model = Model(inputs=inputs, outputs=outputs)

    else:
      deploy_params = self.config['deploy_params']
      outputs = yolov2.post_process(outputs,
                                    deploy_params['iou_threshold'],
                                    deploy_params['score_threshold'],
                                    deploy_params['maximum_boxes'])

      model = Model(inputs=inputs, outputs=outputs)

    model.load_weights(self.config['model']['weight_file'])
    print("Weight file has been loaded in to model")

    return model

  def get_model(self):
    return self.model
