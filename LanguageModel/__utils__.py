from __logger__ import LOGGER_NAME
import logging
from __tokenizer__ import word_tokenizer
from pathlib import Path
from collections import Counter
import time
import json

logger = logging.getLogger(LOGGER_NAME)


def get_vocab(all_tokenized_texts, min_count=None):
    """
    :param all_tokenized_texts (list of list): each item in the parent list is a list of tokens
    :param min_count: threshold for ignoring words on the basis of count
    :return (list): each item is a word to be kept in the vocabulary
    """
    start = time.time()
    all_toks_freq = Counter([tok for doc in all_tokenized_texts for tok in doc])
    logger.info('time taken to extract all tokens = {:.2f} s'.format(time.time() - start))
    vocab = [tok for tok, freq in all_toks_freq.items() if freq > min_count]
    logger.info('ignoring {} words with global count less than {}'.format(len(all_toks_freq) - len(vocab),
                                                                          min_count))
    return vocab


def token_extractor(file_path, min_count=None):
    """
    --> read file at file_path
    --> extract all tokens from this file
    --> ignore words with global minimum count less than threshold
    :param file_path: str path
    :param min_count: threshold for ignoring words on the basis of count
    :return: list of tokens after trimming by params[min_count]
    """
    if not Path(file_path).is_file():
        logger.error('{} is not a valid file'.format(file_path))
        exit(0)

    with open(file_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    all_tok_text = [word_tokenizer.tokenize(item['text']) for item in obj]
    all_text_classes = list(set([item['journal_type'] for item in obj]))
    del obj
    return get_vocab(all_tok_text, min_count=min_count), all_text_classes
