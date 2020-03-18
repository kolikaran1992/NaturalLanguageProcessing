from keras.callbacks import TensorBoard
from __logger__ import LOGGER_NAME
import tensorflow as tf
import os
import json

import logging

logger = logging.getLogger(LOGGER_NAME)


def create_embed_meta(embed_data, path):
    obj = {
        "tensor_name": name,
        "metadata_path": "metadata_{}.tsv".format(name)
    }

    with open(os.path.join("metadata_{}.tsv".format(name)), 'w', encoding='utf-8') as f:
        print(*words, sep='\n', end='', file=f)

    return json.dumps(obj)


class ValidationMetrics(TensorBoard):
    def __init__(self, log_dir='./logs', embed_data={}, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super().__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super().set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super().on_epoch_end(epoch, logs)
