from zoo import darknet19, yolo_preprocess_input
from zoo import mobile_net, preprocess_input
from zoo import densenet

MODEL_ZOO = {'yolov2':    darknet19,
             'mobilenet': mobile_net,
             'densenet':  densenet}


preprocessor = {'yolov2':    yolo_preprocess_input,
                'densenet':  yolo_preprocess_input,
                'mobilenet': preprocess_input}


class FeatureExtractor(object):
    def __init__(self,
                 is_training,
                 img_size,
                 model,
                 model_path=None):

        self.name           = model
        self._is_training   = is_training
        self.img_size       = img_size
        self._preprocess_fn = preprocessor[model]
        self.model          = self._get_feature_extractor_from_zoo(model, img_size, model_path)

    def _get_feature_extractor_from_zoo(self, model, img_size, model_path):
        """
        """
        global MODEL_ZOO
        if model not in MODEL_ZOO:
            raise ValueError("Model is not available in zoo.")

        if model != 'densenet':
            return MODEL_ZOO[model](img_size, include_top=False)
        else:
            return MODEL_ZOO[model](include_top=False, model_path=model_path)

    def preprocess(self, resized_inputs):
        return self._preprocess_fn(resized_inputs)

    def extract_features(self, preprocessed_inputs):
        pass