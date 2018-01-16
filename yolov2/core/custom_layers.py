"""Collections of Custom Keras Layers

See: https://keras.io/layers/writing-your-own-keras-layers/

Example usage:

"""
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
K = tf.keras.backend


class ImageResizer(Layer):
  """Resize image into fixed squared size
  """

  def __init__(self, img_size, **kwargs):
    self.img_size = img_size
    super(ImageResizer, self).__init__(**kwargs)

  def build(self, input_shape):
    super(ImageResizer, self).build(input_shape)

  def call(self, inputs, **kwargs):
    x = tf.image.resize_images(inputs, (self.img_size, self.img_size))
    return x

  def compute_output_shape(self, input_shape):
    input_shape = list(input_shape)
    input_shape[1] = self.img_size
    input_shape[2] = self.img_size
    return tuple(input_shape)

  def get_config(self):
    config = {'img_size': self.img_size}
    base_config = super(ImageResizer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Preprocessor(Layer):
  """Pre-process image before loading into feature extractor
  """

  def __init__(self, pre_process_func, **kwargs):
    self.pre_process_func = pre_process_func
    super(Preprocessor, self).__init__(**kwargs)

  def build(self, input_shape):
    super(Preprocessor, self).build(input_shape)

  def call(self, inputs, **kwargs):
    x = self.pre_process_func(inputs)
    return x

  def compute_output_shape(self, input_shape):
    return tuple(input_shape)

  def get_config(self):
    config = {'pre_process_func': self.pre_process_func}
    base_config = super(Preprocessor, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Reroute(Layer):
  """Reroute layer for YOLOv2
  """

  def __init__(self, block_size=2, **kwargs):
    self.block_size = block_size
    super(Reroute, self).__init__(**kwargs)

  def build(self, input_shape):
    super(Reroute, self).build(input_shape)

  def call(self, inputs, **kwargs):
    return tf.space_to_depth(inputs, self.block_size)

  def compute_output_shape(self, shape):
    block_size = self.block_size
    if shape[1]:
      return tuple([shape[0], shape[1] / block_size, shape[2] / 2, block_size * block_size * shape[-1]])
    else:
      return tuple([shape[0], None, None, block_size * block_size * shape[-1]])

  def get_config(self):
    config = {'block_size': self.block_size}
    base_config = super(Reroute, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#  @TODO: waiting for Tf.repeat() in upcoming TF version
class OutputInterpreter(Layer):
  """
  Convert output features into predictions
  """

  def __init__(self, anchors, num_classes, **kwargs):
    super(OutputInterpreter, self).__init__(**kwargs)
    self.anchors = anchors
    self.num_classes = num_classes

  def build(self, input_shape):
    super(OutputInterpreter, self).build(input_shape)

  def call(self, output_features, **kwargs):
    shape = tf.shape(output_features)
    batch, height, width = shape[0], shape[1], shape[2]

    # ##################
    #  Create offset map
    # ##################
    cx = tf.reshape(tf.tile(tf.range(width), [height]), [-1, height, width, 1])
    cy = tf.tile(tf.expand_dims(tf.range(height), -1), [1, width])
    cy = tf.reshape(cy, [-1, height, width, 1])
    c_xy = tf.to_float(tf.stack([cx, cy], -1))
    # c_xy = tf.reshape(c_xy, [1, 1, 1, tf.shape(c_xy)[0], tf.shape(c_xy)[1]])
    anchors_tensor = tf.to_float(K.reshape(self.anchors, [1, 1, 1, len(self.anchors), 2]))
    output_size = tf.to_float(K.reshape([width, height], [1, 1, 1, 1, 2]))

    outputs = K.reshape(output_features, [batch, height, width, len(self.anchors), self.num_classes + 5])

    # ##################
    # Interpret outputs
    # ##################
    #  (Ref: YOLO-9000 paper)
    box_xy = K.sigmoid(outputs[..., 0:2]) + c_xy
    box_wh = K.exp(outputs[..., 2:4]) * anchors_tensor
    box_confidence = K.sigmoid(outputs[..., 4:5])
    # box_class_probs = K.softmax(outputs[..., 5:])
    # Disabled for compatability tf.softmax_with_logits
    box_class_probs = outputs[..., 5:]

    # Convert coordinates to relative coordinates (percentage)
    box_xy = box_xy / output_size
    box_wh = box_wh / output_size

    # Calculate corner points of bounding boxes
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    # Y1, X1, Y2, X2
    boxes = K.concatenate([box_mins[..., 1:2],
                           box_mins[..., 0:1],  # Y1 X1
                           box_maxes[..., 1:2],
                           box_maxes[..., 0:1]], axis=-1)  # Y2 X2

    outputs = K.concatenate([boxes, box_confidence, box_class_probs], axis=-1)
    return outputs

  def compute_output_shape(self, input_shape):
    return tuple([input_shape[0], input_shape[1], input_shape[2], len(self.anchors), 5 + self.num_classes])

  def get_config(self):
    config = {'anchors': self.anchors,
              'num_classes': self.num_classes}
    base_config = super(OutputInterpreter, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class PostProcessor(Layer):
  """
  Perform Non-Max Suppression to calculate prediction
  """

  def __init__(self, score_threshold, iou_threshold, max_boxes=1000, **kwargs):
    super(PostProcessor, self).__init__(**kwargs)

    self.max_boxes = max_boxes
    self.iou_threshold = iou_threshold
    self.score_threshold = score_threshold

  def build(self, input_shape):
    super(PostProcessor, self).build(input_shape)

  def call(self, inputs, **kwargs):
    boxes = inputs[..., 0:4]
    box_confidence = inputs[..., 4:5]
    box_class_probs = inputs[..., 5:]

    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, -1)
    box_class_scores = K.max(box_scores, -1)
    prediction_mask = (box_class_scores >= self.score_threshold)

    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    nms_index = tf.image.non_max_suppression(boxes,
                                             scores,
                                             max_output_size=self.max_boxes,
                                             iou_threshold=self.iou_threshold)
    boxes = tf.gather(boxes, nms_index)
    scores = K.expand_dims(tf.gather(scores, nms_index), axis=-1)
    classes = K.expand_dims(tf.gather(classes, nms_index), axis=-1)

    return K.concatenate([boxes, K.cast(classes, tf.float32), scores])

  def compute_output_shape(self, input_shape):
    return [(None, 6)]

  def get_config(self):
    config = {'score_threshold': self.score_threshold,
              'iou_threshold': self.iou_threshold,
              'max_boxes': self.max_boxes}
    base_config = super(PostProcessor, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
