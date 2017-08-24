import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw(img, boxes):
    """
    Drawing Bounding Box on Image
    :param img:   
    :param boxes: 
    :return: 
    """
    image = Image.fromarray(np.floor(img).astype('uint8'))
    thickness = (image.size[0] + image.size[1]) // 600
    for box in boxes:
        p1 = (box.x1, box.y1)
        p2 = (box.x2, box.y2)
        label = '{} {:.2f}%'.format(box.cls, box.score * 100)
        draw = ImageDraw.Draw(image)
        # font = ImageFont.truetype(font='./FiraMono-Medium.otf', encoding='ADOB')
        label_size = draw.textsize(label)
        text_origin = np.array([p1[0], p1[1] + 1])
        color = np.random.randint(0, 255, [3])
        color = np.array([0, 255, 0])

        for i in range(thickness):
            draw.rectangle([p1[0] + i, p1[1] + i, p2[0] - i, p2[1] - i], outline=tuple(color))

        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=tuple(color))
        draw.text(tuple(text_origin), label, fill=(0, 0, 0), label_size=2)

        del draw
    return image


class DrawingBox(object):
    def __init__(self, x1, y1, x2, y2, label, score):
        self.x1 = int(x1)
        self.x2 = int(x2)
        self.y1 = int(y1)
        self.y2 = int(y2)
        self.cls = str(label)
        self.score = float(score)


# Test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2

    img = cv2.cvtColor(cv2.imread('../test_images/person.jpg'), cv2.COLOR_BGR2RGB)
    dog = DrawingBox(70, 258, 209, 356, 'Dog', 0.79)
    person = DrawingBox(190, 98, 271, 379, 'Person', 0.81)
    horse = DrawingBox(399, 128, 605, 352, 'Horse', 0.89)
    boxes = [dog, person, horse]

    result = draw(img, boxes)
    plt.figure(figsize=(15, 15))
    plt.imshow(result)
    plt.show()