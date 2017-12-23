"""
Input Pre-processing Methods for MobileNet
"""


def mobilenet_preprocces_func(x):
    x = x / 255.
    x -= 0.5
    x *= 2.
    return x
