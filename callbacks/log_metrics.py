from keras.callbacks import Callback
from __logger__ import LOGGER_NAME

import logging

logger = logging.getLogger(LOGGER_NAME)


class Perplexity(Callback):
    def on_epoch_end(self, epoch, logs={}):
        super(Perplexity, self).on_epoch_end(epoch, logs=logs)
        logger.info(
            'epoch {} end || train_perplexity : {} || val_perplexity : {}'.format(epoch + 1, 2 ** logs['loss'],
                                                                                  2 ** logs['val_loss']))
        logs['perp'] = 2 ** logs['loss']
        logs['val_perp'] = 2 ** logs['val_loss']
