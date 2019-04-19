import os
import random
import argparse
import numpy as np
import scipy.io as sio
from fully_connected import FullyConnected
from utils import empty_list
from multiprocessing import Pool, Process


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=3)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-lr', type=float, default=5e-5)
    parser.add_argument('-lr_decay', type=bool, default=False)
    parser.add_argument('-serial', action='store_true', default=False)
    parser.add_argument('-n_classes', type=int, default=4)
    parser.add_argument('-max_epoches', type=int, default=int(1e6))
    parser.add_argument('-train_data', type=str, default='train_de')
    parser.add_argument('-train_label', type=str, default='train_label_eeg')
    parser.add_argument('-test_data', type=str, default='test_de')
    parser.add_argument('-test_label', type=str, default='test_label_eeg')
    parser.add_argument('-n_processes', type=int, default=4)

    return parser.parse_args()


"""
divide data from this class and from other class into $n$ parts respectively
"""
def decomposition(c):
    data = sio.loadmat('data.mat')
    train_d = data[args.train_data]
    train_l = data[args.train_label]
    test_d = data[args.test_data]
    test_l = data[args.test_label]

    data0 = []
    label0 = []
    data1 = []
    label1 = []

    _data1 = empty_list(args.n)
    _label1 = empty_list(args.n)
    _data0 = empty_list(args.n)
    _label0 = empty_list(args.n)

    for d, l in zip(train_d, train_l):
        if l == c:
            data1.append(d)
            label1.append(1)
        else:
            data0.append(d)
            label0.append(0)

    for d, l in zip(data1, label1):
        r = random.randint(0, args.n - 1)
        _data1[r].append(d)
        _label1[r].append(l)

    for d, l in zip(data0, label0):
        r = random.randint(0, args.n - 1)
        _data0[r].append(d)
        _label0[r].append(l)

    max_labels = []
    for i, (d1, l1) in enumerate(zip(_data1, _label1)):
        min_labels = []
        for j, (d0, l0) in enumerate(zip(_data0, _label0)):
            model = FullyConnected(
                name='%i_%i_%i' % (c, i, j),
                lr=args.lr,
                lr_decay=args.lr_decay,
                n_classes=2,
                max_epoches=args.max_epoches,
                train_data=d1 + d0,
                train_label=l1 + l0,
                test_data=test_d,
                seed=args.seed
            )
            model.train()
            min_labels.append(model.test())
        max_labels.append(np.min(min_labels, axis=0))

    return np.max(max_labels, axis=0)


if not os.path.exists('logs'):
    os.mkdir('logs')
args = parse_arg()


def main():
    if args.serial:
        results = []
        for c in range(args.n_classes):
            results.append(decomposition(c))
    else:
        pool = Pool(args.n_processes)
        results = pool.map(decomposition, range(args.n_classes))

    """ collect """
    test_l = sio.loadmat('data.mat')[args.test_label]
    predicts = np.argmax(results, axis=0)
    acc = np.count_nonzero(test_l[:, 0] + 1 == np.argmax(predicts, axis=0)) / test_l.shape[0]
    print('acc =', acc)


if __name__ == '__main__':
    main()
