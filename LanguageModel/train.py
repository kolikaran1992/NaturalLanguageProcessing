from argparse import ArgumentParser
from pathlib import Path

from keras_applications.nasnet import models

from __logger__ import LOGGER_NAME
import logging
from LanguageModel.text_processor import TextProcessor
from __paths__ import path_to_lm, path_to_language_models
import json
from .model import LanguageModel
from keras.utils import Sequence
import math
import numpy as np
from keras.optimizers import rmsprop
from keras.callbacks import TensorBoard, ModelCheckpoint
from callbacks.log_metrics import Perplexity
from callbacks.tb_scalars import LangModTensorBoard
from time import strftime, gmtime

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

args = parser.parse_args()


class BatchGenerator(Sequence):
    """
    --> yield batches of model input data given all texts
    --> all texts is a list of list, each sublist os a list of tokens
    """

    def __init__(self, all_texts, shuffle=True, batch_size=32, text_transformer=None):
        self.all_texts = all_texts
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._text_transformer = text_transformer

    def __getitem__(self, idx):
        batch_texts = self.all_texts[idx * self.batch_size: (idx + 1) * self.batch_size]
        obj_batch = self._text_transformer.convert_batch(batch_texts, get_outs=True)
        return [obj_batch['words'], obj_batch['chars']], obj_batch['outs']

    def __len__(self):
        return math.ceil(len(self.all_texts) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.all_texts)


def get_saver(name):
    path = path_to_language_models.joinpath(model_name)
    path.mkdir(parents=True, exist_ok=True)
    path = path.joinpath('weights.{epoch:02d}-{val_loss:.2f}.hdf5').as_posix()
    saver = ModelCheckpoint(path, monitor='val_perplexity', verbose=1, save_best_only=False, save_weights_only=True,
                            mode='min', period=1)
    return saver


def get_tb_logs(tb_dir, name):
    if not tb_dir or not tb_dir.is_dir():
        logger.info('tensorboard logs will not be created')
        return

    path = tb_dir.joinpath(name)
    path.mkdir(exists_ok=True, parents=True)
    path.joinpath('{}'.format(format(strftime("%Y-%m-%d %H:%M:%S", gmtime()))))
    tb_logs = LangModTensorBoard(log_dir=path.as_posix())
    return tb_logs


def get_callbacks(name, tb_dir):
    saver = get_saver(name)
    perp_calc = Perplexity()
    tb_logs = get_tb_logs(tb_dir, name)

    return [cb for cb in [saver, perp_calc, tb_logs] if cb is not None]


if __name__ == '__main__':
    args = parser.parse_args()

    file_path = Path(args.file_path.replace("\\", ''))
    model_name = args.model_name
    tensorboard_dir = Path(args.tb_dir) if args.tb_dir else None

    # all_tokens = token_extractor(file_path=file_path, sep='s#e#p#e#r#a#t#o#r', min_count=params.get('min_count'))
    #
    # processor = TextProcessor(max_seq_len=50, max_char_len=15)
    # processor.populate_vocab(all_tokens)
    #
    # processor.print_stats()
    #
    # processor.save('temp')

    p = TextProcessor()
    p.load(model_name)
    lm = LanguageModel(
        word_embedding_size=params.get('word_emb_size'),
        char_embedding_size=params.get('char_emb_size'),
        word_inp_mask_val=p._word_vocab.pad_idx(),
        word_vocab_size=len(p._word_vocab),
        char_vocab_size=len(p._char_vocab),
        max_seq_len=params.get('max_seq_len'),
        max_word_len=params.get('max_token_len'),
        char_cnn_filters=params.get('char_cnn_filters'),
        char_cnn_ker_size=params.get('char_cnn_ker_size'),
        char_cnn_pool_size=params.get('char_cnn_pool_size'),
        dropout=params.get('dropout')
    )
    model = lm.get_model()

    lm.save(model_name)

    lm_2 = LanguageModel()
    lm_2.load(model_name)

    train_gen = BatchGenerator(['I have the power'.split()] * 2, text_transformer=p, batch_size=4)
    val_gen = BatchGenerator(['He man'.split(), 'To be or not to be'.split(),
                              'that is the question'.split(), 'we must be willing to know that we do not know'.split()],
                             text_transformer=p, batch_size=4)

    opt = rmsprop(lr=0.001)
    loss = 'categorical_crossentropy'
    model.compile(optimizer=opt, loss=loss)
    model.summary()

    # prepare callbacks
    all_callbacks = get_callbacks(model_name, tensorboard_dir)

    model.fit_generator(train_gen, steps_per_epoch=len(train_gen),
                        epochs=100, verbose=1, validation_data=val_gen, validation_steps=len(val_gen),
                        callbacks=all_callbacks)

    # obj = p.convert_batch(['I have the power .'.split(' ')]*2)
    # print(obj['words'])
    # print(obj['chars'])
