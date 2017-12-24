"""
Export and Optimize your trained YOLOv2 for interference.

This file will:
    * Reconstruct a clean TF graph
    * Load the trained weights
    * Quantize/Optimize the weights for better performance during interference
    * Convert the trained model into .pb file for running on TF Serving or any other supported platform

In this example, we export YOLOv2 with darknet-19 as feature extractor
"""
from __future__ import print_function

import os
import re
import tensorflow as tf
import keras.backend as K

from keras.layers import Input
from keras.models import Model
from models.yolov2_darknet import yolov2_darknet
from cfg import *

# TF Libraries to export model into .pb file
from tensorflow.python.client import session
from tensorflow.python.framework import graph_io, graph_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.tools.graph_transforms import TransformGraph

import argparse

parser = argparse.ArgumentParser("Export Keras Model to TensorFlow Serving")
parser.add_argument('--output', type=str, default='/tmp/yolov2',
                    help="Export path")
parser.add_argument('--version', type=str, default='1',
                    help="Model Version", )
parser.add_argument('--weight_file', type=str, default=None,
                    help="Path to pre-trained weight files")
parser.add_argument('--iou', type=float, default=0.5,
                    help="IoU value for Non-max suppression")
parser.add_argument('--threshold', type=float, default=0.0,
                    help="Threshold value to display box")


K.set_learning_phase(0)


def _main_():
    # ###############
    # Parse Config  #
    # ###############
    args = parser.parse_args()
    iou               = args.iou
    scores_threshold  = args.threshold
    weight_file       = args.weight_file

    export_dir        = args.output
    model_version     = args.version

    if not os.path.isfile(weight_file):
        raise IOError("Weight file is invalid")

    anchors, class_names = config_prediction()
    # Interference Pipeline for object detection Model
    with K.get_session() as sess:

        inputs = Input(shape=(None, None, 3), name='image_input')
        # #########################
        # Reconstruct Trained Model
        # #########################
        outputs = yolov2_darknet(is_training=False,
                                 inputs=inputs,
                                 img_size=IMG_INPUT_SIZE,
                                 anchors=anchors,
                                 num_classes=N_CLASSES,
                                 iou=iou,
                                 scores_threshold=scores_threshold)

        model = Model(inputs=inputs, outputs=outputs)
        model.load_weights(weight_file)
        model.summary()

        # ########################
        # Configure output Tensors
        # ########################
        outputs = dict()
        outputs['detection_boxes']   = tf.identity(model.output[0], name='detection_boxes')
        outputs['detection_scores']  = tf.identity(model.output[1], name='detection_scores')
        outputs['detection_classes'] = tf.identity(model.output[2], name='detection_classes')

        for output_key in outputs:
            tf.add_to_collection('inference_op', outputs[output_key])

        output_node_names = ','.join(outputs.keys())

        # ###############################
        # Freeze the model into pb format
        # ###############################
        frozen_graph_def = graph_util.convert_variables_to_constants(
                                     sess,
                                     sess.graph.as_graph_def(),
                                     output_node_names.split(','))

        # ####################################
        # Quantize and Optimize Trained Model
        # ####################################
        transforms = ["add_default_attributes",
                      "quantize_weights", "round_weights",
                      "fold_batch_norms", "fold_old_batch_norms"]

        quantized_graph = TransformGraph(frozen_graph_def,
                                         inputs="image_input",
                                         outputs=output_node_names.split(','),
                                         transforms=transforms)

        graph_io.write_graph(quantized_graph, './', 'frozen_graph.pb', as_text=False)

    # #####################
    # Export to TF Serving#
    # #####################
    #   Reference: https://github.com/tensorflow/models/tree/master/research/object_detection

    export_path = os.path.join(export_dir, model_version)
    with tf.Graph().as_default():
        tf.import_graph_def(quantized_graph, name='')

        # Optimizing graph
        rewrite_options = rewriter_config_pb2.RewriterConfig(optimize_tensor_layout=True)
        rewrite_options.optimizers.append('pruning')
        rewrite_options.optimizers.append('constfold')
        rewrite_options.optimizers.append('layout')
        graph_options = tf.GraphOptions(rewrite_options=rewrite_options, infer_shapes=True)

        config = tf.ConfigProto(graph_options=graph_options)
        with session.Session(config=config) as sess:

            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            tensor_info_inputs = {
                'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
            tensor_info_outputs = {}
            for k, v in outputs.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

            detection_signature = (
                    tf.saved_model.signature_def_utils.build_signature_def(
                            inputs     = tensor_info_inputs,
                            outputs    = tensor_info_outputs,
                            method_name= signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                    sess, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={'predict_images': detection_signature,
                                       signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: detection_signature,
                                       },
            )
            builder.save()


def config_prediction():
    # Config Anchors
    anchors = []
    with open(ANCHORS, 'r') as f:
        data = f.read().splitlines()
        for line in data:
            numbers = re.findall('\d+.\d+', line)
            anchors.append((float(numbers[0]), float(numbers[1])))
    # Load class names
    with open(CATEGORIES, mode='r') as txt_file:
        class_names = [c.strip() for c in txt_file.readlines()]
    return anchors, class_names


if __name__ == "__main__":

    _main_()