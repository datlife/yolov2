from PIL import Image, ImageDraw
import numpy as np


def draw_bboxes(img, boxes):
    """
    Drawing Bounding Box on Image
    """
    image = Image.fromarray(np.floor(img).astype('uint8'))
    thickness = (image.size[0] + image.size[1]) // 400
    for box in boxes:
        p1 = (box.x1, box.y1)
        p2 = (box.x2, box.y2)
        label = '{} {:.2f}%'.format(box.cls, box.score * 100)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label)
        text_origin = np.array([p1[0], p1[1] - 10])
        color = [0, 255, 0]

        for i in range(thickness):
            draw.rectangle([p1[0] + i, p1[1] + i, p2[0] - i, p2[1] - i], outline=tuple(color))

        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=tuple(color))
        draw.text(tuple(text_origin), label, fill=(0, 0, 0), label_size=2)

        del draw
    return image
