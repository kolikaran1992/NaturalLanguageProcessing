from argparse import ArgumentParser
from pathlib import Path
from LanguageModel.__logger__ import LOGGER_NAME
import logging
from LanguageModel.vocab import Vocabulary
from LanguageModel.text_processor import TextProcessor
from .__utils__ import token_extractor
from .__paths__ import root
import json

with open(root.joinpath('params.json'), 'r') as f:
    params = json.load(f)

logger = logging.getLogger(LOGGER_NAME)

parser = ArgumentParser()
parser.add_argument("--file_path", "--file_path",
                    dest="file_path", default={}, required=True,
                    help="path to file containing all text")

args = parser.parse_args()

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
    #
    p = TextProcessor()
    p.load('temp')
    obj = p.convert_batch(['I have the power .'.split(' ')]*2)
    print(obj['words'])
    print(obj['chars'])
    print(obj['outs'])