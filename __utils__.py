import json
from __logger__ import LOGGER_NAME
import logging

logger = logging.getLogger(LOGGER_NAME)


def read_json(path):
    if not path.is_file():
        logger.error('json file could not be loaded because "{}" dies not exists'.format(path.as_posix()))
        exit(2)

    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    return obj


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
