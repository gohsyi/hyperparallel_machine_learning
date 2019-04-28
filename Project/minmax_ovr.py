import os
import shutil
import gc
import random
import numpy as np
import scipy.io as sio
import tensorflow as tf
from fully_connected import FullyConnected
from fully_connected_batch import FullyConnectedBatch
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

    d1 = empty_list(args.n)
    l1 = empty_list(args.n)
    d0 = empty_list(args.n)
    l0 = empty_list(args.n)

    for d, l in zip(train_d, train_l):
        r = random.randint(0, args.n - 1)
        if l == c:
            d1[r].append(d)
            l1[r].append(1)
        else:
            d0[r].append(d)
            l0[r].append(0)

    for i, d in enumerate(d1):
        np.savetxt(os.path.join('data', 'ovr_d1_{}_{}.csv'.format(c, i)), d, delimiter=',')
    for i, l in enumerate(l1):
        np.savetxt(os.path.join('data', 'ovr_l1_{}_{}.csv'.format(c, i)), l, delimiter=',')
    for i, d in enumerate(d0):
        np.savetxt(os.path.join('data', 'ovr_d0_{}_{}.csv'.format(c, i)), d, delimiter=',')
    for i, l in enumerate(l0):
        np.savetxt(os.path.join('data', 'ovr_l0_{}_{}.csv'.format(c, i)), l, delimiter=',')


""" 
train the model named $name 
"""
def train(name):
    print('training model %s' % name)
    c, i, j = map(lambda x: int(x), name.split('_'))

    train_d = np.concatenate([
        np.loadtxt(os.path.join('data', 'ovr_d1_{}_{}.csv'.format(c, i)), delimiter=','),
        np.loadtxt(os.path.join('data', 'ovr_d0_{}_{}.csv'.format(c, j)), delimiter=','),
    ])

    train_l = np.concatenate([
        np.loadtxt(os.path.join('data', 'ovr_l1_{}_{}.csv'.format(c, i)), delimiter=','),
        np.loadtxt(os.path.join('data', 'ovr_l0_{}_{}.csv'.format(c, j)), delimiter=','),
    ])

    test_d = sio.loadmat(os.path.join('data', 'data.mat'))[args.test_d]

    if args.batchsize == 0:
        model = FullyConnected(
            folder=folder,
            name=name,
            hidsize=args.hidsize,
            max_epoches=args.max_epoches,
            ac_fn=args.ac_fn,
            lr=args.lr,
            lr_decay=args.lr_decay,
            use_sigmoid=args.sigmoid,
            n_classes=2,
            train_data=train_d,
            train_label=train_l,
            test_data=test_d,
            seed=args.seed)
    else:
        model = FullyConnectedBatch(
            folder=folder,
            name=name,
            batchsize=args.batchsize,
            max_epoches=args.max_epoches,
            hidsize=args.hidsize,
            ac_fn=args.ac_fn,
            lr=args.lr,
            lr_decay=args.lr_decay,
            use_sigmoid=args.sigmoid,
            n_classes=2,
            train_data=train_d,
            train_label=train_l,
            test_data=test_d,
            seed=args.seed)

    model.restore()
    model.train()

    predict = model.predict()
    model.sess.close()

    return predict


args, abstract = parse_arg()
folder = os.path.join('logs', 'ovr_{}'.format(abstract))


def main():
    random.seed(args.n)
    logger = getLogger('logs', 'ovr')
    logger.info(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    for c in range(args.n_classes):
        decomposition(c)

    models = [
        '{}_{}_{}'.format(c, i, j)
        for c in range(args.n_classes)
        for i in range(args.n)
        for j in range(args.n)
    ]

    for iter in range(10):
        """ training """
        tf.reset_default_graph()
        gc.collect()

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

        predicts = np.argmax(max_results, axis=0)
        np.savetxt(os.path.join(folder, 'predicts_{}.csv'.format(iter)), predicts, delimiter=',')

        acc = np.count_nonzero(test_l[:, 0] == predicts) / test_l.shape[0]

        logger.info('ep:{}, acc:{}'.format(args.max_epoches * (iter+1), acc))


if __name__ == '__main__':
    main()
