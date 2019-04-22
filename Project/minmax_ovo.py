import os
import random
import numpy as np
import scipy.io as sio
from fully_connected import FullyConnected
from utils import parse_arg, empty_list, getLogger
from multiprocessing import Pool


"""
divide data from this class and from other class into $n$ parts respectively
"""
def decomposition(c):
    print('preparing data for class %i' % c)
    data = sio.loadmat(os.path.join('data', 'data.mat'))
    train_d = data[args.train_d]
    train_l = data[args.train_l]

    data = empty_list(args.n)

    for d, l in zip(train_d, train_l):
        r = random.randint(0, args.n - 1)
        if l == c:
            data[r].append(d)

    for i, d in enumerate(data):
        np.savetxt(os.path.join('data', 'ovo_d_{}_{}.csv'.format(c, i)), d, delimiter=',')


""" 
train the model named $name 
"""
def train(name):
    print('training model %s' % name)
    c1, c0, i, j = map(lambda x: int(x), name.split('_'))

    d1 = np.loadtxt(os.path.join('data', 'ovo_d_{}_{}.csv'.format(c1, i)), delimiter=',')
    l1 = np.ones(d1.shape[0])
    d0 = np.loadtxt(os.path.join('data', 'ovo_d_{}_{}.csv'.format(c0, j)), delimiter=',')
    l0 = np.zeros(d0.shape[0])

    train_d = np.concatenate([d1, d0])
    train_l = np.concatenate([l1, l0])
    test_d = sio.loadmat(os.path.join('data', 'data.mat'))[args.test_d]

    model = FullyConnected(
        folder=folder,
        name=name,
        hidsz=args.hidsz,
        ac_fn=args.ac_fn,
        lr=args.lr,
        lr_decay=args.lr_decay,
        n_classes=2,
        max_epoches=args.max_epoches,
        train_data=train_d,
        train_label=train_l,
        test_data=test_d,
        seed=args.seed
    )
    model.train()
    return model.classify()


args = parse_arg()

folder = os.path.join('logs', 'ovo_n{}_h{}_{}_lr{}{}ep{}{}'.format(
    args.n,
    args.hidsz,
    args.lr,
    args.ac_fn,
    '_decay_' if args.lr_decay else '_',
    args.max_epoches,
    '_debug' if args.serial else '',
))


def main():
    random.seed(args.n)
    logger = getLogger('logs', 'ovo')
    logger.info(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(folder):
        os.makedirs(folder)

    for c in range(args.n_classes):
        decomposition(c)

    models = [
        '{}_{}_{}_{}'.format(c1, c0, i, j)
        for c1 in range(args.n_classes)
        for c0 in range(c1)
        for i in range(args.n)
        for j in range(args.n)
    ]

    """ training """
    if args.serial:
        results = [train(model) for model in models]
    else:
        pool = Pool(args.n_processes)
        results = pool.map(train, models)

    """ collect """
    test_l = sio.loadmat(os.path.join('data', 'data.mat'))[args.test_l]

    # min
    min_results = []
    for i in range(0, len(results), args.n):
        min_results.append(np.min(results[i:i+args.n], axis=0))

    # max
    max_results = []
    for i in range(0, len(min_results), args.n):
        max_results.append(np.max(min_results[i:i+args.n], axis=0))

    predicts = empty_list(args.n_classes)

    i = 0
    for c1 in range(args.n_classes):
        for c0 in range(c1):
            predicts[c1].append(max_results[i])
            predicts[c0].append(1 - max_results[i])
            i = i + 1

    predicts = np.max(predicts, axis=1)
    predicts = np.argmax(predicts, axis=0)
    predicts.tofile(os.path.join(folder, 'predicts.csv'), sep=',')

    acc = np.count_nonzero(test_l[:, 0] == predicts) / test_l.shape[0]

    logger.info('accuracy = {}'.format(acc))


if __name__ == '__main__':
    main()
