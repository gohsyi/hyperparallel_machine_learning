import os
import sys
import time
import logging
import argparse
from contextlib import contextmanager


"""
argument parser
"""
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=3)
    parser.add_argument('-hidsize', type=str, default='128')
    parser.add_argument('-ac_fn', type=str, default='relu', help='relu/elu/sigmoid/tanh')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-lr_decay', action='store_true', default=False)
    parser.add_argument('-batchsize', type=int, default=64)
    parser.add_argument('-sigmoid', action='store_true', default=False)
    parser.add_argument('-serial', action='store_true', default=False)
    parser.add_argument('-n_classes', type=int, default=4)
    parser.add_argument('-max_epoches', type=int, default=int(1e5))
    parser.add_argument('-train_d', type=str, default='train_de')
    parser.add_argument('-train_l', type=str, default='train_label_eeg')
    parser.add_argument('-test_d', type=str, default='test_de')
    parser.add_argument('-test_l', type=str, default='test_label_eeg')
    parser.add_argument('-n_processes', type=int, default=8)
    parser.add_argument('-gpu', type=str, default='-1')

    args = parser.parse_args()

    abstract = 'n{}_h{}_{}_lr{}{}_bs{}{}_ep{}{}'.format(
        args.n,
        args.hidsize,
        args.ac_fn,
        args.lr,
        '_decay' if args.lr_decay else '',
        args.batchsize,
        '_sigmoid' if args.sigmoid else '',
        args.max_epoches,
        '_debug' if args.serial else '',
    )

    return args, abstract


"""
returns an empty list with shape (d0, d1, d2)
"""
def empty_list(d0, d1=None, d2=None):
    if d1 is None:
        return [[] for _ in range(d0)]
    elif d2 is None:
        return [[[] for _ in range(d1)] for __ in range(d0)]
    else:
        return [[[[] for _ in range(d2)] for __ in range(d1)] for ___ in range(d0)]


"""
returns a logger with std output and file output
"""
def getLogger(folder, name):
    if not os.path.exists(folder):
        os.mkdir(folder)

    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s\tmodel:{}\t%(message)s'.format(name))

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(os.path.join(folder, '{}.log'.format(name)))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


@contextmanager
def timed(msg, logger):
    tstart = time.time()
    yield
    logger.info('%s done in %.3f seconds' % (msg, time.time() - tstart))
