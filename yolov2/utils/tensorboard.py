from keras.callbacks import TensorBoard
import keras.backend as K


class DetectionMonitor(TensorBoard):

    def __init__(self, **kwargs):

        super(DetectionMonitor, self).__init__(**kwargs)
    
    def update(self, generator, steps, global_step):
        self.generator = generator
        self.val_steps = steps
        self.global_step = global_step

    def on_epoch_end(self, epoch, logs=None):

        for i in range(self.val_steps):
            x_batch, y_batch = self.generator.next()
            feed_dict = {self.model.inputs[0]: x_batch,
                         self.model.targets[0]: y_batch,
                         K.learning_phase(): 0.0}  # Testing

            result = self.sess.run([self.merged], feed_dict=feed_dict)
            summary_str = result[0]
            self.writer.add_summary(summary_str, self.global_step)

        self.writer.flush()


