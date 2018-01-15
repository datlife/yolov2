"""
Build YOLOv2 Model

The idea is that YOLOv2 consists of feature extractor and detector.
By using YOLOv2MetaArch, one can swap different types o feature extractor (DarkNet19, MobileNet, NASNet, DenseNet)
and different types of detector, too.

In this file, we construct a standard YOLOv2 using Darknet19 as feature extractor.
"""
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

# @TODO: Multi-scale training
# @TODO: add  ClassificationLoss, Localization, ObjectConfidence
# @TODO: add 10 samples images and draw bounding boxes + ground truths using IoU = 0.5, scores=0.7
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

    # ###############
    # Prepare Data  #
    # ###############
    image_size = self.config['model']['image_size']

    inv_map = {v: k for k, v in self.label_dict.items()}
    inputs, labels = parse_inputs(training_data, inv_map)

    x_train, x_val = train_test_split(inputs, test_size=test_size)
    y_train = [labels[k] for k in x_train]
    y_val   = [labels[k] for k in x_val]

    tfdata = TFData(
      num_classes   = self.num_classes,
      anchors       = self.anchors,
      shrink_factor = self.config['model']['shrink_factor'])

    # ################
    # Compile model  #
    # ################
    objective = YOLOV2Loss(self.anchors, self.num_classes, summary=True)
    self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                       loss=objective.compute_loss)
    # ############################
    # Convert Keras to Estimator #
    # ############################
    # Note: there might be a better solution. I converted to tf.Estimator,
    # instead of using model.fit_generator, so that I can gain more flexibility
    # to play with Tensorboard for visualization purposes
    config = tf.estimator.RunConfig(
      model_dir=None,
      save_summary_steps=20,
      log_step_count_steps=20,
      save_checkpoints_steps=steps_per_epoch,
    )
    estimator = get_estimator(
      model          = self.model,
      custom_objects = custom_objects,
      config         = config,
      params         = self.config,
      label_map      = self.label_dict)

    train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: tfdata.generator(x_train, y_train, image_size, batch_size),
      max_steps= steps_per_epoch*epochs,
    )
    eval_spec = tf.estimator.EvalSpec(
      input_fn=lambda: tfdata.generator(x_val, y_val, image_size, batch_size),
      throttle_secs= 500,
      steps=1
    )
    # ################
    # Start Training #
    # ################
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  def _construct_model(self, is_training, feature_extractor, detector):
    init_weights = None
    if 'based' in self.config['model']['weight_file']:
      init_weights = self.config['model']['weight_file']

    yolov2 = YOLOv2MetaArch(
               feature_extractor=feature_extractor,
               detector=detector,
               anchors=self.anchors,
               num_classes=self.num_classes,
               init_weights= init_weights)

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

    if init_weights:
      model.load_weights(self.config['model']['weight_file'])
    print("Weight file has been loaded in to model")
    return model


# #
# # ########################
# # Start Training Process #
# # ########################
# model.train(
#   input_fn= lambda: tfdata.generator(x_train, y_train, image_size, batch_size),
#   hooks = None,
#   steps = steps_per_epoch
# )
#
# model.evaluate(
#   input_fn= lambda: tfdata.generator(x_val, y_val, image_size, batch_size),
#   hooks = None,
#   steps = int(len(x_val)/batch_size)
# )