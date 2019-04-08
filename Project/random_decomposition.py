import scipy.io as sio
import random
import argparse
import numpy as np
import tensorflow as tf
from fully_connected import FullyConnected
from utils import empty_list
from multiprocessing import Process


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=3)
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-n_classes', type=int, default=4)
    parser.add_argument('-max_epoches', type=int, default=int(1e4))
    parser.add_argument('-train_data', type=str, default='train_de')
    parser.add_argument('-train_label', type=str, default='train_label_eeg')
    parser.add_argument('-test_data', type=str, default='test_de')
    parser.add_argument('-test_label', type=str, default='test_label_eeg')
    args = parser.parse_args()
    return args


"""
divide data from this class and from other class into $n$ parts respectively
"""
def decomposition(args, data1, label1, data0, label0):
    _data1 = empty_list(args.n)
    _label1 = empty_list(args.n)
    _data0 = empty_list(args.n)
    _label0 = empty_list(args.n)

    for d, l in zip(data1, label1):
        r = random.randint(0, args.n-1)
        _data1[r].append(d)
        _label1[r].append(l)

    for d, l in zip(data0, label0):
        r = random.randint(0, args.n - 1)
        _data1[r].append(d)
        _label1[r].append(l)

    models = []
    for d1, l1 in zip(_data1, _label1):
        for d0, l0 in zip(_data0, _label0):
            models.append(FullyConnected(
                name='fc%i' % len(models),
                lr=args.lr,
                n_classes=2,
                max_epoches=args.max_epoches,
                train_data=(d1 + d0),
                train_label=(l1 + l0)))
    return models


def main():
    args = parse_arg()

    data = sio.loadmat('data.mat')
    train_d = data[args.train_data]
    train_l = data[args.train_label]
    test_d = data[args.test_data]
    test_l = data[args.test_label]

    predicts = empty_list(args.n_classes)

    train_data = empty_list(args.n_classes, 2)
    train_label = empty_list(args.n_classes, 2)

    models = []

    for c in range(args.n_classes):
        for d, l in zip(train_d, train_l):
            if l == c:
                train_data[c][0].append(d)
                train_label[c][0].append(1)
            else:
                train_data[c][1].append(d)
                train_label[c][1].append(0)
        with tf.variable_scope('class%i'%c):
            models.append(decomposition(args, train_data[c][0], train_label[c][0], train_data[c][1], train_label[c][1]))

    for c in range(len(models)):
        results = []
        for m in models[c]:
            m.train()
            results.append(m.classify(test_d))
        _results = []
        for i in range(0, len(results), args.n):
            _results.append(np.min(results[i:i+args.n], axis=0))
        results = np.max(_results, axis=0)
        predicts.append(results * c)  # 0->0, 1->c

    acc = np.count_nonzero(test_l == np.argmax(predicts, axis=0))
    print('acc =', acc)


if __name__ == '__main__':
    main()
