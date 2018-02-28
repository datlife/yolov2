"""Convert Keras model to tf.Estimator
"""

import tensorflow as tf
from .net_builder import YOLOv2MetaArch
from tensorflow.python.keras._impl.keras.estimator import _clone_and_build_model, _save_first_checkpoint


def get_estimator(model, custom_objects, model_dir=None, config=None, params=None, label_map=None):
  """Construct a tf.Estimator

  Instead of using fit_generator in keras, one might consider convert a compiled Model
  into Estimator. This enables more flexibility in control during training/evaluation process

  Args:
    model:
    custom_objects:
    model_dir:
    config:
    params:
    label_map:

  Returns:
    `tf.estimator.Estimator` object

  """
  weights = model.get_weights()
  estimator = tf.estimator.Estimator(
    model_fn = _construct_model_fn(model, custom_objects, label_map),
    model_dir= model_dir,
    config   = config,
    params   = params
  )
  _save_first_checkpoint(model, estimator, custom_objects, weights)
  return estimator


# TODO: speed-up training process
def _construct_model_fn(model,  custom_objects, label_map):
  """Convert keras model to tf.Estimator Object
  """
  category_index = {k: {'id': k, 'name': i} for k, i in label_map.items()}

  def model_fn(features, labels, mode, params):
    """See : https://www.tensorflow.org/extend/estimators#constructing_the_model_fn

    Args:
      features: a dict  - passed via `input_fn`
      labels:   a Tensor- passed via `input_fn`
      mode:     tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT)
      params:   a dict - hyper-parameters for training

    Returns:
      `tf.estimator.EstimatorSpec` object
    """
    train_op     = None
    eval_metrics = None
    summaries = []
    # PREDICTION MODE
    if mode is tf.estimator.ModeKeys.PREDICT:
      outputs      = YOLOv2MetaArch.post_process(
        predictions    = model.outputs,
        iou_threshold  = 0.5,
        score_threshold= 0.0,
        max_boxes      = 100)
      inference = tf.keras.models.Model(model.inputs, outputs)
      inference = _clone_and_build_model(mode, inference, custom_objects, features, labels)
      predictions = {
        'detection_boxes': tf.identity(inference.outputs[0], name='detection_boxes'),
        'detection_scores': tf.identity(inference.outputs[1], name='detection_scores'),
        'detection_classes': tf.identity(inference.outputs[2], name='detection_classes')
      }
      return tf.estimator.EstimatorSpec(
         mode = mode,
         predictions= predictions
      )

    # TRAINING AND EVALUATION MODE
    yolo2 = _clone_and_build_model(mode, model, custom_objects, features, labels)
    yolo2._make_train_function()  # pylint: disable=protected-access
    if mode is tf.estimator.ModeKeys.TRAIN:
      train_op = yolo2.train_function.updates_op

    if mode is tf.estimator.ModeKeys.EVAL:
      merged = tf.summary.merge(summaries)

    return tf.estimator.EstimatorSpec(
      mode       = mode,
      loss       = yolo2.total_loss,
      predictions= dict(zip(yolo2.output_names, yolo2.outputs)),
      train_op   = train_op,
      eval_metric_ops=eval_metrics,
    )
  return model_fn