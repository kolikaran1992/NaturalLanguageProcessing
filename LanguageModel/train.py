from argparse import ArgumentParser
from pathlib import Path
from __logger__ import LOGGER_NAME
import logging
from LanguageModel.text_processor import TextProcessor
from __paths__ import path_to_lm
import json
from .model import LanguageModel
from keras.utils import Sequence
import math
import numpy as np
from keras.optimizers import rmsprop

with open(path_to_lm.joinpath('params.json'), 'r') as f:
    params = json.load(f)

logger = logging.getLogger(LOGGER_NAME)

parser = ArgumentParser()
parser.add_argument("--file_path", "--file_path",
                    dest="file_path", default={}, required=True,
                    help="path to file containing all text")

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


if __name__ == '__main__':
    args = parser.parse_args()

    file_path = Path(args.file_path.replace("\\", ''))

    # all_tokens = token_extractor(file_path=file_path, sep='s#e#p#e#r#a#t#o#r', min_count=params.get('min_count'))
    #
    # processor = TextProcessor(max_seq_len=50, max_char_len=15)
    # processor.populate_vocab(all_tokens)
    #
    # processor.print_stats()
    #
    # processor.save('temp')

    p = TextProcessor()
    p.load('temp')
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

    lm.save('temp')

    lm_2 = LanguageModel()
    lm_2.load('temp')

    train_gen = BatchGenerator(['I have the power'.split()] * 2, text_transformer=p, batch_size=4)
    val_gen = BatchGenerator(['He man'.split(), 'To be or not to be'.split(),
                              'that is the question'.split(), 'we must be willing to know that we do not know'.split()],
                             text_transformer=p, batch_size=4)

    opt = rmsprop(lr=0.001)
    loss = 'categorical_crossentropy'
    model.compile(optimizer=opt, loss=loss)
    model.summary()

    model.fit_generator(train_gen, steps_per_epoch=len(train_gen),
                        epochs=100, verbose=1, validation_data=val_gen, validation_steps=len(val_gen))

    # obj = p.convert_batch(['I have the power .'.split(' ')]*2)
    # print(obj['words'])
    # print(obj['chars'])
