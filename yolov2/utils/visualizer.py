import keras.backend as K
import tensorflow as tf
from keras.callbacks import Callback


class TensorBoard(Callback):
  """Tensorboard basic visualizations.

  [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
  is a visualization tool provided with TensorFlow.

  This callback writes a log for TensorBoard, which allows
  you to visualize dynamic graphs of your training and test
  metrics, as well as activation histograms for the different
  layers in your model.

  If you have installed TensorFlow with pip, you should be able
  to launch TensorBoard from the command line:
  ```sh
  tensorboard --logdir=/full_path_to_your_logs
  ```

  Arguments
      log_dir: the path of the directory where to save the log
          files to be parsed by TensorBoard.
      histogram_freq: frequency (in epochs) at which to compute activation
          and weight histograms for the layers of the model. If set to 0,
          histograms won't be computed. Validation data (or split) must be
          specified for histogram visualizations.
      write_graph: whether to visualize the graph in TensorBoard.
          The log file can become quite large when
          write_graph is set to True.
      write_grads: whether to visualize gradient histograms in TensorBoard.
          `histogram_freq` must be greater than 0.
      batch_size: size of batch of inputs to feed to the network
          for histograms computation.
      write_images: whether to write model weights to visualize as
          image in TensorBoard.
      embeddings_freq: frequency (in epochs) at which selected embedding
          layers will be saved.
      embeddings_layer_names: a list of names of layers to keep eye on. If
          None or empty list all the embedding layer will be watched.
      embeddings_metadata: a dictionary which maps layer name to a file name
          in which metadata for this embedding layer is saved. See the
          [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
          about metadata files format. In case if the same metadata file is
          used for all embedding layers, string can be passed.
  """

  def __init__(self, log_dir='./logs',
               histogram_freq=1,
               batch_size=32,
               write_graph=True,
               write_grads=False,
               write_images=False,
               embeddings_freq=0,
               embeddings_layer_names=None,
               embeddings_metadata=None):
    super(TensorBoard, self).__init__()
    self.log_dir = log_dir
    self.histogram_freq = histogram_freq
    self.merged = None
    self.write_graph = write_graph
    self.write_grads = write_grads
    self.write_images = write_images
    self.embeddings_freq = embeddings_freq
    self.embeddings_layer_names = embeddings_layer_names
    self.embeddings_metadata = embeddings_metadata or {}
    self.batch_size = batch_size

  def build_summary(self, model, summaries):
    self.model = model
    self.sess = K.get_session()
    model_summaries = set()
    for layer in self.model.layers[2:]:
      model_summaries.add(tf.summary.histogram('{}_out'.format(layer.name), layer.output))

    summaries |= model_summaries
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    self.merged = summary_op

    self.writer = tf.summary.FileWriter(self.log_dir + '/training', )
    self.val_writer = tf.summary.FileWriter(self.log_dir + '/validation')

    if self.write_graph:
      self.writer.add_graph(K.get_session().graph)

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}

    if not self.validation_data and self.histogram_freq:
      raise ValueError('If printing histograms, validation_data must be '
                       'provided, and cannot be a generator.')
    if self.validation_data and self.histogram_freq:
      if epoch % self.histogram_freq == 0:

        val_data = self.validation_data
        tensors = (self.model.inputs +
                   self.model.targets +
                   self.model.sample_weights)

        if self.model.uses_learning_phase:
          tensors += [K.learning_phase()]

        assert len(val_data) == len(tensors)
        val_size = val_data[0].shape[0]
        i = 0
        while i < val_size:
          step = min(self.batch_size, val_size - i)
          if self.model.uses_learning_phase:
            # do not slice the learning phase
            batch_val = [x[i:i + step] for x in val_data[:-1]]
            batch_val.append(val_data[-1])
          else:
            batch_val = [x[i:i + step] for x in val_data]
          assert len(batch_val) == len(tensors)
          feed_dict = dict(zip(tensors, batch_val))
          result = self.sess.run([self.merged], feed_dict=feed_dict)
          summary_str = result[0]
          self.val_writer.add_summary(summary_str, epoch)
          i += self.batch_size

    for name, value in logs.items():
      print(name, value)
      if name in ['batch', 'size']:
        continue
      summary = tf.Summary()
      summary_value = summary.value.add()
      summary_value.simple_value = value.item()
      summary_value.tag = name
      if name == 'val_loss':
        self.val_writer.add_summary(summary, epoch)
      elif name == 'loss':
        self.writer.add_summary(summary, epoch)

    self.writer.flush()
    self.val_writer.flush()

  def on_train_end(self, _):
    self.writer.close()
