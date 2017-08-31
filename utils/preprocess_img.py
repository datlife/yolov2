"""
Pre-process Image Input
"""

def preprocess_img(img):
    img = img/255.
    # img -= 0.5
    # img *= 2
    return img