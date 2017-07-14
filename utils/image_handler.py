import cv2
import numpy as np


def random_transform(img, bbox):
    """
    Augment image randomly
    """
    a = np.random.randint(0, 2, [1, 3]).astype('bool')[0]
    aug_box = bbox
    if a[0] == 1:
        img, aug_box = shear(img, bbox)
    if a[1] == 1:
        img = blur(img)
    if a[2] == 1:
        img = brightness(img)
    # GAN can apply here to create random occlusion and deformation

    return img, aug_box


def update_box(bbox, rotation_matrix):
    """
    Update bounding accordingly to the rotation matrix
    :param bbox: [[(x1,y1), (x2, y2)]]
    :param rotation_matrix: 
    :return: 
    """
    # Transform to correct bounding box when performing augmentation
    rotated_box = []
    if bbox is not None:
        for box in bbox:
            for pts in box:
                pts = np.asarray(pts).astype('float32')  # Convert to float
                pts = np.append(pts, [0.], 0)            # 3-D : (x, y, z)
                pts = np.expand_dims(pts, axis=1)

                # new_box = cv2.transform([pts], rotation_matrix)
                new_box = np.dot(rotation_matrix, pts).T.flatten().astype(int).tolist()
                rotated_box.append(tuple(new_box))
    return [rotated_box]


def shear(img, bbox=None):
    " Affirm Transformation orignal image"
    x, y, channel = img.shape
    shear = np.random.randint(3, 7)
    pts1 = np.array([[5, 5], [20, 5], [5, 20]]).astype('float32')

    pt1 = 5 + shear * np.random.uniform() - shear / 2
    pt2 = 20 + shear * np.random.uniform() - shear / 2

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    M = cv2.getAffineTransform(pts1, pts2)
    result = cv2.warpAffine(img, M, (y, x))
    rotated_box = update_box(bbox, M)

    return result, rotated_box


def blur(img):
    r_int = np.random.randint(0, 6)
    odd_size = 2 * r_int + 1
    return cv2.GaussianBlur(img, (odd_size, odd_size), 0)


def brightness(img):
    gamma = np.random.uniform(0.4, 1.7)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_img = cv2.LUT(img, table)
    return new_img


def rotate(img, bbox=None):
    """
        Rotate original image at random rotation angle from -30 degree to +30 degree
    :param img: 
    :param bbox: 
    :return: 
    """

    row, col, channel = img.shape
    angle = np.random.uniform(-30, 30)
    rotation_point = (row / 2, col / 2)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (col, row))

    # Update bounding box
    rotated_box = update_box(bbox, rotation_matrix)
    return rotated_img, rotated_box


def preprocess_img(img):
    img = img/255
    # img -= 0.5
    # img *= 2
    return img


def draw_boxes(img, boxes, color=(0, 255, 0), thickness=6):
    draw_img = np.copy(img)
    for box in boxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thickness)
    return draw_img

