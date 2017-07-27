"""
Implementation of Hierarchical Soft-Max Tree

Input:
-----
lisa.names
"""
import numpy as np


class Node(object):
    name = ' '
    index = -1
    parent = None
    children = None
    is_leaf  = False


class SoftMaxTree(object):
    def __init__(self, tree_file):
        self.tree = None

    def update_probability(self, softmax_vector):
        """
        Return a new soft-max vector by level of tree
        :param softmax_vector:
        :return:
        """
        raise NotImplemented

    def get_hierachy_probability(self, key):
        raise NotImplemented

if __name__ == "__main__":
    tree = []