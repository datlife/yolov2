import cv2
import numpy as np
import time
import config as cfg

import multiprocessing as mp
from threading import Thread

from yolov2.utils import parse_config
from yolov2.utils import draw, draw_fps
from yolov2.tfserving.client import ObjectDetectionClient

# Command line arguments
import argparse
parser = argparse.ArgumentParser(description="Webcam demo")

parser.add_argument('--server', type=str, default='localhost:9000',
                    help="PredictionService host:port [default=localhost:9000]")

parser.add_argument('--model', type=str, default='yolov2',
                    help="tf serving model name [default=yolov2]")

parser.add_argument('--video_source', type=int, default=0,
                    help="Source to webcam [default = 0]")


def main():
    # ############
    # Parse Config
    # ############
    ARGS = parser.parse_args()
    _, label_dict = parse_config(cfg)

    # #########
    # Init Demo
    # ##########
    object_detector = ObjectDetectionClient(ARGS.server, ARGS.model, label_dict, verbose=True)
    video_capture = WebcamVideoStream(ARGS.video_source).start()

    # ##########
    # Start Demo
    # ###########
    viewer = WebCamViewer(video_capture, object_detector)
    print("Initialized object detection client at {} with model {}".format(ARGS.server, ARGS.model))

    viewer.run()


class WebCamViewer(object):
    def __init__(self, video_capture, detector):
        self.video    = video_capture
        self.detector = detector

        self.input_q  = mp.Queue(maxsize=5)
        self.output_q = mp.Queue(maxsize=1)
        pool = mp.Pool(processes=mp.cpu_count()-1, initializer=self.worker)

    def run(self):
        boxes = []
        classes = []
        scores = []
        num_frames = 0
        detection_fps = 0
        detection_frames = 0

        start = time.time()
        while True:
            frame = self.video.read()

            if self.input_q.full():
                self.input_q.get()

            self.input_q.put(frame)

            if not self.output_q.empty():
                detection_frames += 1
                detection_fps = detection_frames / (time.time() - start)

                boxes, classes, scores = self.output_q.get()

            num_frames += 1
            camera_fps = num_frames / (time.time() - start)
            frame = draw_fps(frame, camera_fps, detection_fps)
            frame = draw(frame, boxes, classes, scores)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elapsed = (time.time() - start)
        print('[INFO] elapsed time (total): {:.2f}'.format(elapsed))
        print('[INFO] approx. FPS: {:.2f}'.format(num_frames / elapsed))
        print('[INFO] approx. detection FPS: {:.2f}'.format(detection_frames / elapsed))
        self.video.stop()
        cv2.destroyAllWindows()

    def worker(self):
        while True:
            frame = self.input_q.get()
            self.output_q.put(self.detect_objects_in(frame))

    def detect_objects_in(self, frame, threshold=0.7):
        h, w, _ = frame.shape

        boxes, classes, scores = self.detector.predict(frame)

        # Filter out results that is not reach a threshold
        filtered_result = [(b, c, s/100.) for b, c, s in zip(boxes, classes, scores) if s/100. > threshold]
        boxes, classes, scores = zip(*filtered_result)
        boxes = [box * np.array([h, w, h, w]) for box in boxes]
        return boxes, classes, scores


class WebcamVideoStream(object):
    def __init__(self, src):
        # initialize the video camera stream
        self.stream = cv2.VideoCapture(src)

        # set to highest frame rate
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        # @TODO: might change to MJPEG later

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


if __name__ == "__main__":
    main()
