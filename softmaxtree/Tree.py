"""
Implementation of Hierarchical Soft-Max Tree

Input:
-----
lisa.tree
"""
import numpy as np
import tensorflow as tf
import keras.backend as K
from cfg import N_CLASSES, N_ANCHORS
# Show a structure of a tree
# Each node has the following [name, parent] - index is inferred from the order of this list


class Node(object):

    def __init__(self, id, node_name, parent, height=0):
        self.id       = id
        self.name     = node_name
        self.parent   = parent
        self.children = []
        self.height   = height

    def __str__(self):
        str = "%s - %s \n " % (self.id, self.name)
        for child in self.children:
            for i in range(0, self.height, 1):
                str += "\t"
            str += "\t%s" % child
        return str


class SoftMaxTree(object):

    def __init__(self, tree_file='lisa.tree'):
        """
        Construct Soft Max tree
        :param tree_file:
        """
        fl = open(tree_file)
        self.tree_dict = {}
        self.is_built = False
        try:
            lines = [l for l in fl.read().splitlines() if l is not '']  # Filter empty line and comment

            # Create a root node
            root_name = lines[0].split(',')[0]
            self.tree_dict[0] = Node(id=0, node_name=root_name, parent=None)

            for idx, line in enumerate(lines[1:]):
                idx = idx + 1
                name, parent_id = ["".join(s.split()) for s in line.split(',')]  # strip all whitespaces
                print(name, parent_id)
                height   = self.tree_dict[int(parent_id)].height + 1

                # Create new node and add to graph dictionary
                new_node = Node(idx, name, self.tree_dict[int(parent_id)], height)
                self.tree_dict[idx] = new_node                            # Add a new node into dictionary
                self.tree_dict[int(parent_id)].children.append(new_node)  # Update Children

        except Exception as e:
            print("Error while building tree", e)
        finally:
            fl.close()

        if len(self.tree_dict) == len(lines):
            self.is_built = True
        else:
            raise RuntimeError("There were error while constructing Tree. Please check tree file format.")

    def encode_label(self, index):
        '''
        Create a encoded binary vector for a class
            Example: SpeedLimit45 is a subset of SpeedLimit, which is a subset of Traffic Sign

            TrafficSign       SpeedLimit    Speed45
        [0. 0. 1. 0. 0. 0. ... 1...... 0. ...1... 0. ...]

        args
        ----
           index : an integer

        return : an binary_encoded label vector
        ------

        '''
        encoded_label = np.eye(len(self.tree_dict))[index]

        # Enable parent node
        parent_id = self.tree_dict[index].parent.id
        encoded_label[parent_id] = 1.0
        return encoded_label

    def calculate_softmax(self, idx, logits, labels, pred_obj_conf, true_obj_conf):
        """
        Update Probabilities of each labels accordingly to Hierarchical Structure
        :param idx:   default = -1 / starting from root of soft-max tree
        :param logits:
        :param labels:
        :return:
        """
        loss = 0.0
        gt_shape = tf.shape(labels)[1:3]
        GRID_W, GRID_H = gt_shape[0], gt_shape[1]

        if idx == 0:
            # Object Confidence Loss
            weight_conf = 0.5 * (1. - true_obj_conf) + 5.0 * true_obj_conf
            loss = tf.pow(true_obj_conf - pred_obj_conf, 2) * weight_conf
            loss = tf.reshape(loss, [-1, tf.cast(GRID_W * GRID_H, tf.int32) * N_ANCHORS])
            loss = tf.reduce_mean(tf.reduce_sum(loss, 1))

        if self.tree_dict[idx].children:   # traverse until reaching the leaf
            weights_prob = tf.concat(len(self.tree_dict[idx].children) * [true_obj_conf], 4)
            first_child = self.tree_dict[idx].children[0].id
            logits_softmax = logits[..., first_child:first_child + len(self.tree_dict[idx].children)]
            labels_softmax = labels[...,  first_child:first_child + len(self.tree_dict[idx].children)]

            # Calculate Parent Probability
            if idx == 0:
                logits_softmax = pred_obj_conf * tf.nn.softmax(logits_softmax)
            else:
                parent = self.tree_dict[idx].parent.children[0].id
                parent_softmax = tf.nn.softmax(logits[..., parent: parent + len(self.tree_dict[idx].parent.children)])

                for i, child in enumerate(self.tree_dict[idx].parent.children):
                    if child.id == idx:
                        parent_prob = pred_obj_conf * parent_softmax[..., i:i + 1]
                        # parent_prob    = parent_softmax[..., i:i + 1]

                        logits_softmax = parent_prob * tf.nn.softmax(logits_softmax)

            sub_loss = tf.pow(labels_softmax - logits_softmax, 2) * weights_prob
            sub_loss = tf.reshape(sub_loss, [-1, GRID_H * GRID_W * N_ANCHORS * (len(self.tree_dict[idx].children))])
            sub_loss = tf.reduce_mean(tf.reduce_sum(sub_loss, 1))
            loss += sub_loss

            # Calculate loss of each children
            for children in self.tree_dict[idx].children:
                sub_loss = self.calculate_softmax(idx=children.id, logits=logits, labels=labels,
                                                  pred_obj_conf=pred_obj_conf, true_obj_conf=true_obj_conf)
                loss += sub_loss
        return loss

# Test
if __name__ == "__main__":
    tree = SoftMaxTree(tree_file='../dataset/combined_lisa/lisa.tree')
    print(tree.tree_dict[0])
