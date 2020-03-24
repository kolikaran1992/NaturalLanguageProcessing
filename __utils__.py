import json
from __logger__ import LOGGER_NAME
import logging
from argparse import ArgumentTypeError
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

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
        raise ArgumentTypeError('Boolean value expected.')


def _plot_hist(data, cumulative=False):
    # sns.distplot(token_lengths, norm_hist=True)
    plt.hist(data, cumulative=cumulative, linewidth=1, edgecolor='black',
             weights=np.ones(len(data)) / len(data))
    plt.title('Sequence Length Distribution', fontdict={'size': 20})
    plt.xlabel('Senquence Length', fontdict={'size': 15})
    plt.ylabel('{}Percentage'.format('Cumulative ' if cumulative else ''), fontdict={'size': 15})
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.grid(True, which='both')


def save_hist(data, save_path):
    plt.figure(figsize=(15, 5))
    cumulative = [False, True]
    for i in range(2):
        plt.subplot(1, 2, i+1)
        _plot_hist(data, cumulative=cumulative[i])
    plt.savefig(save_path.joinpath('seq_len_dist.png'))
    logger.info('sequence length distribution saved at {}'.format(save_path.joinpath('seq_len_dist.png')))
