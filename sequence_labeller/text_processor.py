from LanguageModel.vocab import Vocabulary
from __logger__ import LOGGER_NAME
import logging

logger = logging.getLogger(LOGGER_NAME)


class OutputProcessor(object):
    def __init__(self, max_seq_len=50, max_tok_len=15):
        self._tag_vocab = Vocabulary()
        self._max_seq_len = max_seq_len
        self._max_tok_len = max_tok_len

    def populate_vocab(self, all_outputs):
        self._tag_vocab = Vocabulary()
        self._tag_vocab.build(all_outputs, ['<unk>', '<pad>', '<end>'])
        logger.info('successfully built tag vocabulary')
