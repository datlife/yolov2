from FeatureExtractor import FeatureExtractor


def _test():
    darknet = FeatureExtractor(is_training=True, img_size=None, model='darknet19')
    darknet.model.summary()

    darknet = FeatureExtractor(is_training=True, img_size=None, model='mobilenet')
    darknet.model.summary()

    darknet = FeatureExtractor(is_training=True, img_size=None, model='densenet')
    darknet.model.summary()


if __name__ == '__main__':
    _test()
