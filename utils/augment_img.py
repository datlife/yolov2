import numpy as np
import cv2
import copy
from utils.box import Box


def augment_img(img, objects):
    aug_img      = np.copy(img)
    copy_objects = copy.deepcopy(objects)
    HEIGHT, WIDTH, c = img.shape
    # scale the image
    scale = np.random.uniform() / 10. + 1.
    aug_img = cv2.resize(aug_img, (0, 0), fx=scale, fy=scale)

    # translate the image
    max_offx = (scale - 1.) * WIDTH
    max_offy = (scale - 1.) * HEIGHT
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    aug_img = aug_img[offy: (offy + HEIGHT), offx: (offx + WIDTH)]

    # # Changing color
    aug_img = change_brightness(aug_img)

    # fix object's position and size
    new_labels = []
    for obj in copy_objects:
        gt_box, gt_label = obj
        xc, yc, w, h = gt_box.to_array()
        xc = int(xc * scale - offx)
        xc = max(min(xc, WIDTH), 0)
        yc = int(yc * scale - offy)
        yc = max(min(yc, HEIGHT), 0)
        new_labels.append([Box(xc, yc, w*scale, h*scale), gt_label])

    return aug_img, new_labels


def change_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = 0.5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2]*random_bright
    image1[:, :, 2][image1[:, :, 2] > 255]  = 255

    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def random_crop_and_rescale(img, objects):
    """

    :param img:
    :param objects:
    :return:
    """

    # Look for all object
    xmin
