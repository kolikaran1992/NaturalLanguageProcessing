from argparse import ArgumentParser
from pathlib import Path
from LanguageModel.__logger__ import LOGGER_NAME
import logging
from LanguageModel.vocab import token_extractor, Vocabulary

logger = logging.getLogger(LOGGER_NAME)

parser = ArgumentParser()
parser.add_argument("--file_path", "--file_path",
                    dest="file_path", default={}, required=True,
                    help="path to file containing all text")

args = parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()

    file_path = Path(args.file_path.replace("\\", ''))

    all_tokens = token_extractor(file_path=file_path)

    vocab = Vocabulary()
    vocab.build(all_tokens)
    print(len(vocab))
