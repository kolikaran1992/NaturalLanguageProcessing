from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from .__utils__ import get_vocab
from .__logger__ import LOGGER_NAME
import logging
from .vocab import Vocabulary
from .__paths__ import path_to_models, root
import json

logger = logging.getLogger(LOGGER_NAME)

with open(root.joinpath('params.json'), 'r') as f:
    params = json.load(f)


class TextProcessor(object):
    """
    --> Transforms text input (list of tokens) to model input format
    """

    def __init__(self,
                 max_seq_len=None,
                 max_char_len=None
                 ):
        self._word_vocab = None
        self._char_vocab = None

        ## params
        self._max_seq_len = max_seq_len
        self._max_char_len = max_char_len

    def populate_vocab(self, all_tokens):
        """
        --> populate char and word vocabularies
        --> resets word, char vocabs
        :param all_tokens: list of all tokens to be kept in vocabulary
        :return: None
        """
        self._word_vocab = Vocabulary()
        self._char_vocab = Vocabulary()

        self._word_vocab.build(all_tokens)
        logger.info('successfully built word vocabulary')

        all_chars = list(set([ch for tok in all_tokens for ch in tok]))
        self._char_vocab.build(all_chars)
        logger.info('successfully built char vocabulary')

    def save(self, name):
        """
        --> saves meta data to "LanguageModel/saved_models/name/vocab_meta.pkl"
        --> saves word vocab to "LanguageModel/saved_models/name/vocab_word.pkl"
        --> saves char vocab to "LanguageModel/saved_models/name/vocab_char.pkl"
        :param name (str): name for saving the object
        :return: None
        """
        path_to_vocab = path_to_models.joinpath(name)
        path_to_vocab.mkdir(parents=True, exist_ok=True)

        with open(path_to_vocab.joinpath("vocab_meta.json"), 'w', encoding='utf-8') as f:
            all_items = {k: v for k, v in self.__dict__.items() if k not in ['_word_vocab', '_char_vocab']}
            json.dump(all_items, f)

        self._word_vocab.save(path_to_vocab.joinpath('vocab_word.json'))
        self._char_vocab.save(path_to_vocab.joinpath('vocab_char.json'))

        logger.info('text transformer saved successfully at {}'.format(path_to_vocab))

    def load(self, name):
        """
        --> loads meta data from "LanguageModel/saved_models/name/vocab_meta.pkl"
        --> loads word vocab from "LanguageModel/saved_models/name/vocab_word.pkl"
        --> loads char vocab from "LanguageModel/saved_models/name/vocab_char.pkl"
        :param name: name for loading the object
        :return: None
        """
        path_to_vocab = path_to_models.joinpath(name)
        if not path_to_vocab.is_dir():
            logger.error('vocab object could not be loaded since "" is not a valid directory'.format(path_to_vocab))
            exit(1)

        with open(path_to_vocab.joinpath("vocab_meta.json"), 'r', encoding='utf-8') as f:
            obj = json.load(f)
            for k, v in obj.items():
                self.__setattr__(k, v)

        self._word_vocab = Vocabulary()
        self._word_vocab.load(path_to_vocab.joinpath('vocab_word.json'))

        self._char_vocab = Vocabulary()
        self._char_vocab.load(path_to_vocab.joinpath('vocab_char.json'))

        logger.info('text transformer loaded successfully from {}'.format(path_to_vocab))

    def get_stats(self):
        return {
            'word_vocab': len(self._word_vocab),
            'char_vocab': len(self._char_vocab),
            'max_seq_length': self._max_seq_len,
            'max_word_length': self._max_char_len
        }

    def print_stats(self):
        print('Word Vocab : {}\nChar Vocab: {}\nMAX_SEQ_LENGTH: {}\nMAX_TOKEN_LENGTH: {}'.format(
            len(self._word_vocab),
            len(self._char_vocab),
            self._max_seq_len,
            self._max_char_len
        ))

    def _get_outs(self, batch_toks):
        """
        --> get output for language model for a single example
        --> algo: for each item in batch
                --> get all toks except first
                --> append self._word_vocab.end_idx() to the above list
                pad the sequences to self._max_seq_len
        :param batch_toks (list of list): each sublist is a list of tokens
        :return: 2d numpy array
        """
        all_outputs = []
        sequences = [self._word_vocab.to_idx(tokens[1:] + ['<end>']) for tokens in batch_toks]
        _temp = pad_sequences(sequences, maxlen=self._max_seq_len, dtype=params.get('dtype_int'),
                              padding='post',
                              truncating='post',
                              value=self._word_vocab.pad_idx())
        for seq in _temp:
            _t = to_categorical(seq, num_classes=len(self._word_vocab))
            all_outputs.append(_t)
        all_outputs = np.array(all_outputs)
        return all_outputs

    def _convert_single_example(self, toks):
        """
        --> transforms list of tokens to vocab indices
        :param toks: list of tokens
        :return (list, 2d np.array): (word vocab idxs, padded char vocab idxs)
                                    => shape of 2d np.array = (self._max_seq_len, self._max_char_len)
        """
        ## convert words to list of char idxs
        char_seqs = [self._char_vocab.to_idx(tok) for tok in toks]
        ## add padding idxs to match max_seq_len
        char_seqs += [[self._char_vocab.pad_idx()]] * (self._max_seq_len - len(toks))

        padded_char_idxs = pad_sequences(char_seqs,
                                         maxlen=self._max_char_len, padding='post', truncating='post',
                                         value=self._char_vocab.pad_idx(),
                                         dtype=params.get('dtype_int'))

        return self._word_vocab.to_idx(toks), padded_char_idxs

    def convert_batch(self, batch_toks, get_outs=True):
        """
        --> run self._convert_single_example for each item in batch
        --> pad word idxs
        :param get_outs (boolean): True if output for language model needs to be calculated
        :param batch_toks: list of list
                          --> Each sublist is a list of tokens
        :return (2d np array, 3d np array): (padded toks, padded chars)
        """
        all_tok_inps = []
        all_char_inps = []

        for toks in batch_toks:
            toks, chars = self._convert_single_example(toks)
            all_tok_inps.append(toks)
            all_char_inps.append(chars)

        padded_word_inps = pad_sequences(all_tok_inps, maxlen=self._max_seq_len, dtype=params.get('dtype_int'),
                                         padding='post',
                                         truncating='post',
                                         value=self._word_vocab.pad_idx())

        outs = self._get_outs(batch_toks) if get_outs else None

        return {'words': padded_word_inps, 'chars': np.array(all_char_inps), 'outs': outs}
