import json
from .__logger__ import LOGGER_NAME
import logging

logger = logging.getLogger(LOGGER_NAME)


class Vocabulary(object):
    """
    --> builds vocabulary
    --> functionality:
                    => token to and fro idx, append '<unk>' and '<pad>' tokens to existing list of words
                    => save/load vocabulary
    """

    def __init__(self):
        self._vocab2int = None
        self._int2vocab = None

    def __len__(self):
        return len(self._vocab2int)

    def __getitem__(self, item):
        if self._vocab2int is None:
            logger.error('trying to access elements from empty vocabulary')
            return None
        return self._int2vocab[item] if not isinstance(item, str) else self._vocab2int[
            item] if item in self._vocab2int else self._vocab2int['<unk>']

    def build(self, all_toks):
        """
        --> resets vocabulary (vocab2int, int2vocab)
        --> filters tokens by min count
        --> adds unk, pad token
        :param all_words: list of tokens
        :return: None
        """
        extras = ['<unk>', '<pad>', '<end>']
        self._vocab2int = {tok: idx for idx, tok in enumerate(all_toks + extras)}
        self._int2vocab = {idx: tok for idx, tok in enumerate(all_toks + extras)}

    def to_idx(self, tokens):
        """
        --> converts tokens to indices
        --> if token does not exists in the vocab, then the corresponding idx will be that of <unk>
        :param tokens: list of tokens
        :return: list of indices
        """
        return [self._vocab2int[tok] if tok in self._vocab2int else self._vocab2int['<unk>'] for tok in tokens]

    def to_toks(self, indices):
        """
        --> converts ints to tokens
        :param indices: list of indices
        :return: list of tokens
        """
        return list(map(lambda x: self._int2vocab[x], indices))

    def pad_idx(self):
        """
        --> returns pad index
        :return: int
        """
        return self._vocab2int['<pad>']

    def end_idx(self):
        """
        --> returns pad index
        :return: int
        """
        return self._vocab2int['<end>']

    def save(self, path):
        """
        :param path: exact path where vocab needs to be saved
        :return: None
        """
        vocab = self._vocab2int
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f)
        logger.info('vocabulary saved successfully at {}'.format(path))

    def load(self, path):
        if not path.is_file():
            logger.error(
                'vocabulary could not be loaded because it does not exists at the provided path "{}"'.format(path))
            exit(0)
        with open(path, 'r', encoding='utf-8') as f:
            vocab2int = json.load(f)

        self.__setattr__('_vocab2int', {tok: int(idx) for tok, idx in vocab2int.items()})
        self.__setattr__('_int2vocab', {int(idx): tok for tok, idx in vocab2int.items()})

        logger.info('vocabulary loaded successfully from {}'.format(path))
