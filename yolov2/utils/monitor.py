import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import TensorBoard


class DetectorCallback(TensorBoard):

    def __init__(self, **kwargs):
        super(DetectorCallback, self).__init__(**kwargs)
        self.generator   = None
        self.steps       = 32
        self.global_step = 0
        self.model       = None
        self.sess        = None

        self.train_writer = None
        self.val_writer   = None

    def set_model(self, model):
        if self.model is None:
            self.model = model
            self.sess   = K.get_session()

            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.log_dir + "/training", graph=self.sess.graph)
            self.val_writer   = tf.summary.FileWriter(self.log_dir + "/validation")

    def on_epoch_end(self, epoch, logs=None):

        for i in range(self.steps):
            images, labels = self.generator.next()

            feed_dict = {self.model.inputs[0]: images,
                         self.model.targets[0]: labels,
                         K.learning_phase(): 0.0}

            result = K.get_session().run([self.merged], feed_dict)
            self.writer.add_summary(result[0], self.global_step)

        for tag, value in logs.items():
            report = tf.Summary()
            sum_op = report.value.add()
            sum_op.simple_value = value
            sum_op.tag = tag
            self.writer.add_summary(report, self.global_step)

        self.writer.flush()

    def update(self, generator, steps, global_step):
        self.generator = generator
        self.steps     = steps
        self.global_step = global_step

