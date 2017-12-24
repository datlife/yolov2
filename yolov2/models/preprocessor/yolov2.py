"""
Input Pre-processing Methods for DarkNet 19
"""


def yolov2_preprocess_func(inputs):
    inputs = inputs / 255.
    return inputs
