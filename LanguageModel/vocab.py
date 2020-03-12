import json
from .__logger__ import LOGGER_NAME
import logging
from .__paths__ import root
from .__tokenizer__ import word_tokenizer
from pathlib import Path
from collections import Counter
import time

with open(root.joinpath('params.json'), 'r') as f:
    params = json.load(f)

logger = logging.getLogger(LOGGER_NAME)


class Vocabulary(object):
    """
    --> builds vocabulary
    --> can add words
    """

    def __init__(self):
        self._vocab2int = {}
        self._int2vocab = {}

    def build(self, all_toks):
        """
        --> resets vocabulary (vocab2int, int2vocab)
        --> filters tokens by min count
        --> adds unk, pad token
        :param all_words: list of tokens
        :return: None
        """
        extras = ['<unk>', '<pad>']
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

    def save(self, path):
        vocab = self._vocab2int
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab)
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

    def __len__(self):
        return len(self._vocab2int)


def token_extractor(file_path, sep='\n'):
    """
    --> read file at file_path
    --> extract all tokens from this file
    --> ignore words with global minimum count less than threshold
    :param file_path: str path
    :return: list of tokens after trimming by params[min_count]
    """
    min_count = params.get('min_count')

    if not Path(file_path).is_file():
        logger.error('{} is not a valid file'.format(file_path))
        exit(0)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    ## del when input text is fixed
    text = text.replace('s#e#p#e#r#a#t#o#r', ' ')

    start = time.time()
    all_toks_freq = Counter([tok for tok in word_tokenizer.tokenize(text)])
    logger.info('time taken to extract all tokens = {:.2f} s'.format(time.time() - start))
    vocab = [tok for tok, freq in all_toks_freq.items() if freq > min_count]
    logger.info('ignoring {} words with global count less than {}'.format(len(all_toks_freq) - len(vocab),
                                                                          min_count))
    return vocab
