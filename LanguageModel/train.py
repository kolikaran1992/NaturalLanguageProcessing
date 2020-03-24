from argparse import ArgumentParser
from pathlib import Path

from __logger__ import LOGGER_NAME
import logging
from LanguageModel.training_wrapper import TrainWrapper
from __paths__ import path_to_lm, path_to_language_models
import json
from keras.utils import Sequence
import math
import numpy as np
from keras.optimizers import rmsprop
from keras.callbacks import TensorBoard, ModelCheckpoint
from callbacks.metric_to_keras_logs import Perplexity
from callbacks.tensorboard import ValidationMetrics
from time import strftime, gmtime
from sklearn.model_selection import train_test_split
from __tokenizer__ import word_tokenizer

with open(path_to_lm.joinpath('params.json'), 'r') as f:
    params = json.load(f)

logger = logging.getLogger(LOGGER_NAME)

parser = ArgumentParser()
parser.add_argument("--file_path", "--file_path",
                    dest="file_path", default={}, required=True,
                    help="path to file containing all text")

parser.add_argument("--model_name", "--model_name",
                    dest="model_name", default='temp', required=True,
                    help="name used for saving language model")

parser.add_argument("--tb_dir", "--tb_dir",
                    dest="tb_dir", default=None, required=False,
                    help="tensorboard logs dir")

parser.add_argument("--retrain", "--retrain",
                    dest="retrain", default=None, required=False,
                    help="if 1 the model will be retrained from last saved wts")

args = parser.parse_args()


class BatchGenerator(Sequence):
    """
    --> yield batches of model input data given all texts and their corresponding classes
    --> all texts is a list of list, each sublist os a list of tokens
    """

    def __init__(self, all_texts, all_classes, shuffle=True, batch_size=32, text_transformer=None):
        self.all_texts = all_texts
        self.all_classes = all_classes
        self._idxs = list(range(len(all_texts)))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._text_transformer = text_transformer

    def __getitem__(self, idx):
        batch_idxs = self._idxs[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_texts = [self.all_texts[i] for i in batch_idxs]
        batch_cls = [self.all_classes[i] for i in batch_idxs]

        obj_batch = self._text_transformer.convert_batch(batch_texts, batch_cls, get_outs=True)
        return [obj_batch['words'], obj_batch['chars'], obj_batch['class']], obj_batch['outs']

    def __len__(self):
        return math.ceil(len(self.all_texts) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._idxs)


def get_saver(name, save_pd):
    path = path_to_language_models.joinpath(name)
    path.mkdir(parents=True, exist_ok=True)
    path = path.joinpath('weights.{epoch:02d}-{val_loss:.2f}.h5').as_posix()
    saver = ModelCheckpoint(path, monitor='val_perplexity', verbose=1, save_best_only=False, save_weights_only=True,
                            mode='min', period=save_pd)
    return saver


def get_tb_logs(tb_dir, name):
    if not tb_dir or not tb_dir.is_dir():
        logger.info('tensorboard logs will not be created')
        return

    path = tb_dir.joinpath(name)
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath('{}'.format(format(strftime("%Y-%m-%d %H:%M:%S", gmtime()))))
    tb_logs = ValidationMetrics(log_dir=path.as_posix())
    return tb_logs


def get_callbacks(name, tb_dir, save_pd):
    saver = get_saver(name, save_pd)
    perp_calc = Perplexity()
    tb_logs = get_tb_logs(tb_dir, name)

    return [cb for cb in [saver, perp_calc, tb_logs] if cb is not None]


def get_train_val_idxs(length, val_size, random_state):
    train_idxs, val_idxs, _, _ = train_test_split(
        [idx for idx in range(length)], [None] * length, test_size=val_size,
        random_state=random_state)
    logger.info('total train samples = {}'.format(len(train_idxs)))
    logger.info('total validation samples = {}'.format(len(val_idxs)))
    return train_idxs, val_idxs


if __name__ == '__main__':
    args = parser.parse_args()

    file_path = Path(args.file_path)
    model_name = args.model_name
    tensorboard_dir = Path(args.tb_dir) if args.tb_dir else None
    retrain = True if int(args.retrain) == 1 else False
    batch_size = 16

    wrapper = TrainWrapper(
        model_name=model_name,
        retrain=retrain,
        texts_path=file_path
    )

    model, text_processor = wrapper.get_model()

    with open(file_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)[:100]

    train_idxs, val_idxs = get_train_val_idxs(len(obj), 0.2, 37)

    train_gen = BatchGenerator([word_tokenizer.tokenize(obj[i]['text']) for i in train_idxs],
                               [obj[i]['journal_type'] for i in train_idxs],
                               text_transformer=text_processor,
                               batch_size=batch_size)
    val_gen = BatchGenerator([word_tokenizer.tokenize(obj[i]['text']) for i in val_idxs],
                             [obj[i]['journal_type'] for i in val_idxs],
                             text_transformer=text_processor, batch_size=batch_size)

    del obj, train_idxs, val_idxs


    opt = rmsprop(lr=0.001)
    loss = 'categorical_crossentropy'
    model.compile(optimizer=opt, loss=loss)
    model.summary()

    # prepare callbacks
    all_callbacks = get_callbacks(model_name, tensorboard_dir, 1)

    model.fit_generator(train_gen, steps_per_epoch=len(train_gen),
                        epochs=100, verbose=0, validation_data=val_gen, validation_steps=len(val_gen),
                        callbacks=all_callbacks)
