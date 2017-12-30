import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer


class ImageResizer(Layer):
    """
    Resize image into fixed squared size before loading into feature extractor
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
    """
    Pre-process image before loading into feature extractor
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
    """
    Reroute layer for YOLOv2
    """
    def __init__(self, block_size=2,  **kwargs):
        self.block_size        = block_size
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
        config = {'block_size':self.block_size}
        base_config = super(Reroute, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PostProcessor(Layer):
    """
    Perform Non-Max Suppression to calculate prediction
    """
    def __init__(self, score_threshold, iou_threshold, max_boxes=1000, **kwargs):
        self.max_boxes         = max_boxes
        self.iou_threshold     = iou_threshold
        self.score_threshold   = score_threshold

        super(PostProcessor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PostProcessor, self).build(input_shape)

    def call(self, inputs, **kwargs):

        boxes           = inputs[..., 0:4]
        box_confidence  = inputs[..., 4:5]
        box_class_probs = inputs[..., 5:]

        box_scores       = box_confidence * box_class_probs
        box_classes      = K.argmax(box_scores, -1)
        box_class_scores = K.max(box_scores, -1)
        prediction_mask  = (box_class_scores >= self.score_threshold)

        boxes   = tf.boolean_mask(boxes, prediction_mask)
        scores  = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)
        nms_index = tf.image.non_max_suppression(boxes,
                                                 scores,
                                                 max_output_size=self.max_boxes,
                                                 iou_threshold=self.iou_threshold)
        boxes   = tf.gather(boxes, nms_index)
        scores  = K.expand_dims(tf.gather(scores, nms_index), axis=-1)
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
