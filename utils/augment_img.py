import cv2
import numpy as np
import math


def preprocess_img(img):
    img = img/255.
    img -= 0.5
    img *= 2
    return img


def random_transform(img, bbox):
    """
    Augment image randomly
    """
    a = np.random.randint(0, 2, [1, 3]).astype('bool')[0]
    aug_box = bbox
    # if a[0] == 1:
    #     img, aug_box = rotate(img, bbox)
    if a[1] == 1:
        img = blur(img)
    # if a[2] == 1:
    #     img = change_brightness(img)
    # GAN can apply here to create random occlusion and deformation

    return img, aug_box


def change_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5+np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2]*random_bright
    image1[:, :, 2][image1[:, :, 2] > 255]  = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def blur(img):
    r_int = np.random.randint(0, 2)
    odd_size = 2 * r_int + 1
    return cv2.GaussianBlur(img, (odd_size, odd_size), 0)


def rotate(img, bbox=None):
    """
        Rotate original image at random rotation angle from -30 degree to +30 degree
    :param img: 
    :param bbox: 
    :return: 
    """
    row, col, channel = img.shape
    angle = np.random.uniform(-20, 20)
    rotation_point = (row / 2, col / 2)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (col, row))

    # Update bounding box
    angle = angle * (np.pi / 180)
    rotated_box =[]
    for box in bbox:
        for pts in box:
            pts = np.asarray(pts, dtype=np.float)
            x = pts[0]
            y = pts[1]
            axis_x = x - rotation_point[0]
            axis_y = y - rotation_point[1]

            x = axis_x * math.cos(angle) + axis_y * math.sin(angle)
            y = (-axis_x) * math.sin(angle) + axis_y * math.cos(angle)
            x = x + rotation_point[0]
            y = y + rotation_point[1]
            rotated_box.append(tuple(np.array([x, y])))
    return rotated_img, [rotated_box]



