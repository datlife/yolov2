from FeatureExtractor import FeatureExtractor


def _test():
    darknet = FeatureExtractor(is_training=True, img_size=None, model='yolov2')
    darknet.model.summary()

    darknet = FeatureExtractor(is_training=True, img_size=None, model='mobilenet')
    darknet.model.summary()

    darknet = FeatureExtractor(is_training=True, img_size=None, model='densenet')
    darknet.model.summary()


if __name__ == '__main__':
    _test()
