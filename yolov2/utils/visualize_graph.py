"""
Visualize mode in Tensorboard
"""
import sys
import tensorflow as tf
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2


def visualize_graph_in_tfboard(filename, output='./log'):
    with tf.Session() as sess:
        model_filename = filename
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)
            if 1 != len(sm.meta_graphs):
                print('More than one graph found. Not sure which to write')
                sys.exit(1)

            g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)

        train_writer = tf.summary.FileWriter(output)
        train_writer.add_graph(sess.graph)
        print("Please execute `tensorboard --logdir {}` to view graph".format(output))

