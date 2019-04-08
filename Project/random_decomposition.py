import scipy.io as sio
import random
import argparse
from fully_connected import FullyConnected
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


def main():
    args = parse_arg()

    data = sio.loadmat('data.mat')
    train_d = data[args.train_data]
    train_l = data[args.train_label]
    test_d = data[args.test_data]
    test_l = data[args.test_label]

    m = train_d.shape[0]

    train_data = [[[],[]] for _ in range(args.n_classes)]
    train_label = [[[],[]] for _ in range(args.n_classes)]

    for c in range(args.n_classes):
        for i in range(m):
            if train_l[i] == c:
                train_data[c][0].append(train_d[i])
                train_label[c][0].append(0.9)
            else:
                train_data[c][1].append(train_d[i])

    models = []
    for i in range(args.n):
        models.append(FullyConnected(
            name='fc%i'%i,
            lr=args.lr,
            max_epoches=args.max_epoches,
            train_data=train_data[i],
            train_label=train_label[i]))

    for model in models:
        model.train()




if __name__ == '__main__':
    main()