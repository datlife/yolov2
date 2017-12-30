"""
Implementation of Hierarchical Soft-Max Tree

Input:
-----
lisa.tree
"""
import numpy as np


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

        Args
           index : an integer

        return :
            An encoded label vector
        '''
        encoded_label = np.eye(len(self.tree_dict))[index]
        # Set parent node to 1.0
        parent_id = self.tree_dict[index].parent.id
        encoded_label[parent_id] = 1.0
        return encoded_label


# Test
if __name__ == "__main__":
    tree = SoftMaxTree(tree_file='/home/dat/Documents/yolov2/dataset/combined_lisa/lisa.tree')
    print(tree.tree_dict[0])
