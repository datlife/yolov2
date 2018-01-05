"""
Build YOLOv2 Model

The idea is that YOLOv2 consists of feature extractor and detector.
By using YOLOv2MetaArch, one can swap different types o feature extractor (DarkNet19, MobileNet, NASNet, DenseNet)
and different types of detector, too.

In this file, we construct a standard YOLOv2 using Darknet19 as feature extractor.
"""
import numpy as np
from sklearn.model_selection import train_test_split

import keras
from keras.layers import Input
from keras.models import Model

from yolov2.core.loss import YOLOV2Loss
from yolov2.core.net_builder import YOLOv2MetaArch

from yolov2.utils.generator import TFData
from yolov2.utils.parser import parse_inputs, parse_label_map


class YOLOv2(object):

    def __init__(self, is_training, feature_extractor, detector, config_dict):

        self.is_training = is_training
        self.config      = config_dict

        # @TODO: remove 608.
        self.anchors     = np.array(config_dict['anchors']) / (608. / 32)
        self.num_classes = config_dict['model']['num_classes']
        self.label_dict   = parse_label_map(config_dict['label_map'])

        self.model       = self._construct_model(is_training, feature_extractor, detector)

    def train(self, training_data, epochs, batch_size, learning_rate):

        # ###############
        # Compile model #
        # ###############
        loss  = YOLOV2Loss(self.anchors, self.num_classes)
        model = self.model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                                   loss=loss.compute_loss)

        # ###############
        # Prepare Data  #
        # ###############
        inv_map = {v: k for k, v in self.label_dict.iteritems()}
        inputs, labels = parse_inputs(training_data, inv_map)

        tfdata = TFData(num_classes, anchors, shrink_factor)
        for current_epoch in range(epochs):
            # Create 10-fold split
            x_train, x_val = train_test_split(inputs, test_size=0.2)
            y_train = [labels[k] for k in x_train]
            y_val = [labels[k] for k in x_val]

            model.fit_generator(generator=tfdata.generator(x_train, y_train, image_size, batch_size),
                                steps_per_epoch=1000,
                                validation_data=tfdata.generator(x_val, y_val, image_size, batch_size),
                                validation_steps=100,
                                verbose=1,
                                workers=0)

            # @TODO: Evaluate y_val and save summaries to TensorBoard

    def _construct_model(self, is_training, feature_extractor, detector):

        yolov2 = YOLOv2MetaArch(feature_extractor= feature_extractor,
                                detector         = detector,
                                anchors          = self.anchors,
                                num_classes      = self.num_classes)
        inputs  = Input(shape=(None, None, 3), name='input_images')
        outputs = yolov2.predict(inputs)

        if is_training:
            model = Model(inputs=inputs, outputs=outputs)

        else:
            deploy_params = self.config['deploy_params']
            outputs = yolov2.post_process(outputs,
                                          deploy_params['iou_threshold'],
                                          deploy_params['score_threshold'],
                                          deploy_params['maximum_boxes'])
            model = Model(inputs=inputs, outputs=outputs)

        model.load_weights(self.config['model']['weight_file'])
        print("Weight file has been loaded in to model")

        return model

