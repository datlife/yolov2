from FeatureExtractor import FeatureExtractor


def _test():
    darknet = FeatureExtractor(is_training=True, img_size=None, model='yolov2')
    darknet.model.summary()

    mobilenet = FeatureExtractor(is_training=True, img_size=None, model='mobilenet')
    mobilenet.model.summary()

    densenet = FeatureExtractor(is_training=True, img_size=None, model='densenet',
                                model_path='../weights/feature_extractor/densenet201.h5')
    densenet.model.summary()


if __name__ == '__main__':
    _test()
