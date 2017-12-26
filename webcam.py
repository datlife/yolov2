import cv2
import time
import config as cfg
import tensorflow as tf
from multiprocessing import Queue, Pool

from yolov2.utils import parse_config
from yolov2.utils import draw, draw_fps
from yolov2.utils import WebcamVideoStream

from yolov2.tfserving.client import ObjectDetectionClient

# Command line arguments
tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_string('model', 'ssd', 'tf serving model (yolov2, ssd, fasterrcnn')
tf.app.flags.DEFINE_string('src', 0, "Source to webcam [default = 0]")

FLAGS = tf.app.flags.FLAGS

model = FLAGS.model
_, label_dict = parse_config(cfg)

detector = ObjectDetectionClient('localhost:9000', model, label_dict)


def main(_):
    video_capture = WebcamVideoStream(FLAGS.src).start()

    input_q = Queue(maxsize=5)
    output_q = Queue(maxsize=1)
    pool = Pool(1, worker, (input_q, output_q))

    boxes   = []
    classes = []
    scores  = []

    num_frames       = 0
    detection_fps    = 0
    detection_frames = 0

    start = time.time()
    while True:
        frame = video_capture.read()

        if input_q.full():
            input_q.get()

        input_q.put(frame)

        if not output_q.empty():
            boxes, classes, scores = output_q.get()
            detection_frames += 1
            detection_fps = detection_frames / (time.time() - start)

        num_frames += 1
        camera_fps = num_frames / (time.time() - start)

        # frame = draw(frame, boxes, classes, scores)
        frame = draw_fps(frame, camera_fps, detection_fps)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    elapsed = (time.time() - start)
    print('[INFO] elapsed time (total): {:.2f}'.format(elapsed))
    print('[INFO] approx. FPS: {:.2f}'.format(num_frames / elapsed))
    print('[INFO] approx. detection FPS: {:.2f}'.format(detection_frames / elapsed))
    video_capture.stop()
    cv2.destroyAllWindows()


def worker(input_q, output_q):
    while True:
        frame = input_q.get()
        output_q.put(detect_objects_in(frame))


def detect_objects_in(frame):
    global detector
    data = detector.predict(frame)
    boxes, classes, scores = filter_out(threshold=0.8 , data=data)
    return boxes, classes, scores


def filter_out(threshold, data):
    boxes, classes, scores = data
    new_boxes = []
    new_classes = []
    new_scores = []
    for b, c, s in zip(boxes, classes, scores):
        if s > threshold:
            new_boxes.append(b)
            new_classes.append(c)
            new_scores.append(s)
    return new_boxes, new_classes, new_scores

if __name__ == "__main__":
    tf.app.run()
