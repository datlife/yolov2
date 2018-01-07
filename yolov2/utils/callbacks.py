import os
import tensorflow as tf


def create_callbacks(backup_dir, tfboard_dir='./logs'):
    # Set up callbacks
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print("A backup directory has been created")

    tf_board = tf.keras.callbacks.TensorBoard(log_dir       = tfboard_dir,
                                           histogram_freq= 0,
                                           write_graph   = False,
                                           write_images  = False)

    backup = tf.keras.callbacks.ModelCheckpoint(os.path.join(backup_dir,
                                             "best_%s-{epoch:02d}-{val_loss:.2f}.weights" % 'darknet19'),
                                             monitor          = 'val_loss',
                                             save_weights_only= True,
                                             save_best_only   = True)

    return [tf_board, backup]
