from argparse import ArgumentParser
from __logger__ import LOGGER_NAME
import logging
from __paths__ import path_to_saved_vocab
from time import time
from .tokenizer import get_tokens
from collections import Counter
from pathlib import Path
from __utils__ import read_json, str2bool, save_seq_len_dist_hist, save_json
from .vocab import Vocabulary

logger = logging.getLogger(LOGGER_NAME)

parser = ArgumentParser()
parser.add_argument("--file_path", "--file_path", type=str,
                    dest="file_path", default={}, required=True,
                    help="path to json file containing all text")

parser.add_argument("--name", "--name",
                    dest="name", required=True, type=str,
                    help="name used for saving vocabulary")

parser.add_argument("--lower", "--lower",
                    dest="lower", default=None, required=False, type=str2bool,
                    help="if True the text will be lower cased")

parser.add_argument("--ascii_only", "--ascii_only",
                    dest="ascii_only", default=False, required=False, type=str2bool,
                    help="if True only ascii tokens (all chars ascii) will be kept in the vocabulary")

parser.add_argument("--tf_thresh", "--tf_thresh",
                    dest="tf_thresh", default=0.01, required=False, type=float,
                    help="all tokens below tf_thresh term frequency percentage in the corpus will be excluded from "
                         "the vocabulary")

args = parser.parse_args()


def collect_tokens(path, to_lower, keep_ascii_only, thresh):
    """
    --> collect word, char and filetype tokens
    :return: word_vocab, char_vocab, filetype_vocab
    """
    obj = read_json(path)

    word_toks, char_toks, filetype_toks, seq_lens = [], [], [], []
    check_ascii = lambda x: all(ord(c) < 128 for c in x)

    for item in obj:
        tokens = [tok for tok in get_tokens(item['text'], lower=to_lower) if check_ascii(tok) and keep_ascii_only]
        seq_lens.append(len(tokens))
        filetype_toks.append(item['filetype'])
        word_toks += tokens
        char_toks += [ch for tok in tokens for ch in tok]

    meta = {'word_tokens': len(word_toks), 'char_tokens': len(char_toks), 'file_type_tokens': len(filetype_toks),
            'total_sequences': len(obj)}

    word_vocab = [tok for tok, count in Counter(word_toks).items() if count > thresh]
    char_vocab = [ch for ch, count in Counter(char_toks).items() if count > thresh]
    file_type_vocab = [ft for ft, count in Counter(filetype_toks).items() if count > thresh]

    for size, total_size, n in zip(map(len, [word_vocab, char_vocab, file_type_vocab]),
                                   map(lambda x: len(set(x)), [word_toks, char_toks, filetype_toks]),
                                   ['word', 'char', 'file_type']):
        logger.info(
            'term frequency thresh "{}" leaves {} tokens out of {} in {} vocabulary'.format(thresh, size, total_size,
                                                                                            n))
    return word_vocab, char_vocab, file_type_vocab, seq_lens, meta


def save_vocab(tokens, ex, p):
    vocab = Vocabulary()
    vocab.build(tokens, ex)
    vocab.save(p)


if __name__ == '__main__':
    args = parser.parse_args()

    file_path = Path(args.file_path)
    vocab_name = args.name
    lower = args.lower
    ascii_only = args.ascii_only
    term_freq_thresh = args.tf_thresh

    logger.info('non ascii tokens will{}be ignored'.format(' ' if ascii_only else ' not '))
    logger.info('text will{}be lower cased'.format(' ' if lower else ' not '))

    if not vocab_name:
        logger.warning('no vocab name provided, setting vocab name to "temp"')
        vocab_name = 'temp'

    start = time()
    words, chars, file_types, sequence_lengths, vocab_meta = collect_tokens(file_path, lower, ascii_only,
                                                                            term_freq_thresh)
    logger.info('time taken to collect tokens = {} s'.format(time() - start))

    vocab_meta['lower'] = lower
    vocab_meta['ascii_only'] = ascii_only
    vocab_meta['term_freq_thresh'] = term_freq_thresh

    path = path_to_saved_vocab.joinpath(vocab_name)
    path.mkdir(exist_ok=True, parents=True)

    save_seq_len_dist_hist(sequence_lengths, path)

    for toks, extras, name in zip([words, chars, file_types], [['<unk>', '<pad>', '<end>']] * 2 + [['<unk>', '<pad>']],
                                  ['word', 'char', 'file_type']):
        save_path = path.joinpath(name + '.json')
        save_vocab(toks, extras, save_path)

    # save vocabulary metadata
    save_json(vocab_meta, path.joinpath('metadata.json'))
    logger.info('vocabulary metadata saved at "{}"'.format(path.joinpath('metadata.json').as_posix()))
