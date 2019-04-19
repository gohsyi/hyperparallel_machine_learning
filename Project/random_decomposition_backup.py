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
def decomposition(args, data1, label1, data0, label0, c):
    _data1 = empty_list(args.n)
    _label1 = empty_list(args.n)
    _data0 = empty_list(args.n)
    _label0 = empty_list(args.n)

    for d, l in zip(data1, label1):
        r = random.randint(0, args.n - 1)
        _data1[r].append(d)
        _label1[r].append(l)

    for d, l in zip(data0, label0):
        r = random.randint(0, args.n - 1)
        _data1[r].append(d)
        _label1[r].append(l)

    model_names = []
    train_data = []
    train_label = []

    for i, (d1, l1) in enumerate(zip(_data1, _label1)):
        for j, (d0, l0) in enumerate(zip(_data0, _label0)):
            model_names.append('%i_%i_%i' % (c, i, j))
            train_data.append(d1 + d0)
            train_label.append(l1 + l0)

    return model_names, train_data, train_label


g_model_names = []
g_train_data = []
g_train_label = []
g_test_data = []


"""
train our models serially
"""
def train(args):
    pass


"""
train only one model, return test labels
"""
def train_one_model(args, name, train_data, train_label, test_data):
    model = FullyConnected(
        name=name,
        lr=args.lr,
        lr_decay=args.lr_decay,
        n_classes=2,
        max_epoches=args.max_epoches,
        train_data=train_data,
        train_label=train_label,
        test_data=test_data,
        seed=args.seed
    )
    model.train()
    return model.test()


"""
train our models in parallel
"""
def train_in_parallel(args):
    pool = Pool(processes=args.n_processes)
    pool.map(train_one_model, zip(g_model_names, g_train_data, g_train_label, g_test_data))


def main():
    if not os.path.exists('logs'):
        os.mkdir('logs')

    args = parse_arg()

    data = sio.loadmat('data.mat')
    train_d = data[args.train_data]
    train_l = data[args.train_label]
    test_d = data[args.test_data]
    test_l = data[args.test_label]

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
        name, d, l = decomposition(
            c=c,
            args=args,
            data1=train_data[c][0],
            label1=train_label[c][0],
            data0=train_data[c][1],
            label0=train_label[c][1],
        )

        g_model_names.append(name)
        g_train_data.append(d)
        g_train_label.append(l)

    """ training """
    if args.serial:
        train(args)
    else:
        train_in_parallel(args)

    """ predicting """
    predicts = []
    for c in range(args.n_classes):
        results = []
        for m in models[c]:
            results.append(m.classify(test_d))
        _results = []
        for i in range(0, len(results), args.n):
            _results.append(np.min(results[i:i+args.n], axis=0))
        results = np.max(_results, axis=0)
        predicts.append(results * (c+1))  # 0->0, 1->(c+1)

    acc = np.count_nonzero(test_l[:, 0] + 1 == np.argmax(predicts, axis=0)) / test_l.shape[0]
    print('acc =', acc)


if __name__ == '__main__':
    main()
