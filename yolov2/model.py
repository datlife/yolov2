"""
Build YOLOv2 Model

The idea is that YOLOv2 consists of feature extractor and detector.
By using YOLOv2MetaArch, one can swap different types o feature extractor (DarkNet19, MobileNet, NASNet, DenseNet)
and different types of detector, too.

In this file, we construct a standard YOLOv2 using Darknet19 as feature extractor.
"""
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from sklearn.model_selection import train_test_split

from yolov2.core.loss import YOLOV2Loss
from yolov2.core.estimator import get_estimator
from yolov2.core.net_builder import YOLOv2MetaArch
from yolov2.core.custom_layers import Preprocessor, PostProcessor, OutputInterpreter

from yolov2.utils.generator import TFData
from yolov2.utils.parser import parse_inputs, parse_label_map

K = tf.keras.backend

custom_objects = {
  'Preprocessor': Preprocessor,
  'PostProcessor': PostProcessor,
  'OutputInterpreter': OutputInterpreter,
}


class YOLOv2(object):

  def __init__(self, is_training, feature_extractor, detector, config_dict):
    """Constructor

    Args:
      is_training:
      feature_extractor:
      detector:
      config_dict:
    """
    self.config = config_dict
    self.is_training = is_training

    self.anchors     = np.array(config_dict['anchors'])
    self.num_classes = config_dict['model']['num_classes']
    self.label_dict  = parse_label_map(config_dict['label_map'])

    self.model   = self._construct_model(is_training, feature_extractor, detector)

  def train(self, training_data, epochs, steps_per_epoch, batch_size, learning_rate, test_size=0.2):
    """

    Args:
      training_data:
      epochs:
      steps_per_epoch:
      batch_size:
      learning_rate:
      test_size:

    Returns:

    """
    # ###############
    # Prepare Data  #
    # ###############
    inverse_dict = {v: k for k, v in self.label_dict.items()}
    inputs, labels = parse_inputs(training_data, inverse_dict)

    # ################
    # Compile model  #
    # ################
    objective = YOLOV2Loss(self.anchors, self.num_classes)
    self.model.compile(optimizer= tf.keras.optimizers.Adam(lr=learning_rate),
                       loss     = objective.compute_loss)

    # ################
    # Start training #
    # ################
    img_sizes = np.array(self.config['model']['image_size'])
    tfdata = TFData(self.num_classes, self.anchors, self.config['model']['shrink_factor'])
    for i in range(epochs):
      x_train, x_val = train_test_split(inputs, test_size=test_size, shuffle=True)
      y_train  = [labels[k] for k in x_train]
      img_size = random.choice(img_sizes)

      train_gen = tfdata.generator(x_train, y_train, img_size, batch_size, shuffle=True)
      for step in range(steps_per_epoch):
        x_batch, y_batch = train_gen.next()
        loss = self.model.train_on_batch(x_batch, y_batch)

      # Eval
      y_val    = [labels[k] for k in x_val]
      val_gen   = tfdata.generator(x_val, y_val, img_size, batch_size, shuffle=True)

  def _construct_model(self, is_training, feature_extractor, detector):
    yolov2 = YOLOv2MetaArch(
               feature_extractor=feature_extractor,
               detector=detector,
               anchors=self.anchors,
               num_classes=self.num_classes)

    inputs  = Input(shape=(None, None, 3), name='input_images')
    outputs = yolov2.predict(inputs)
    if is_training:
      model = Model(inputs=inputs, outputs=outputs)
    else:
      deploy_params = self.config['deploy_params']
      outputs = yolov2.post_process(
                  outputs,
                  deploy_params['iou_threshold'],
                  deploy_params['score_threshold'],
                  deploy_params['maximum_boxes'])

      model = Model(inputs=inputs, outputs=outputs)

    model.load_weights(self.config['model']['weight_file'])
    print("Weight file has been loaded in to model")
    return model

# ############################
# Convert Keras to Estimator #
# ############################
# config = tf.estimator.RunConfig(
#   model_dir=None,
#   save_summary_steps=20,
#   log_step_count_steps=20,
#   save_checkpoints_steps=steps_per_epoch,
# )
#
# estimator = get_estimator(
#   model=self.model,
#   custom_objects=custom_objects,
#   config=config,
#   params=self.config,
#   label_map=self.label_dict)
