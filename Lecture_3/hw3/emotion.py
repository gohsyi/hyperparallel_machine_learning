import sys
sys.path.append('/usr/local/3rdparty/libsvm/python/')

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from svmutil import *


def svm_scale(x, lower, upper):
    return (x-np.min(x)) * (upper-lower) / (np.max(x)-np.min(x)) + lower


def scale(train_data, test_data):
    feature_min = np.min(train_data)
    feature_max = np.max(train_data)

    train_data = svm_scale(train_data, -1, 1)
    test_data = svm_scale(test_data, -np.min(test_data)/feature_min, np.max(test_data)/feature_max)

    return train_data, test_data


def visualize(train_data, train_label, test_data, test_label):
    tsne = TSNE(n_components=2)
    # _data = tsne.fit_transform(np.concatenate([train_data, test_data]))

    pca = PCA(n_components=2)
    pca.fit(train_data)
    _data = pca.transform(np.concatenate([train_data, test_data]))

    scatter_x = [[], [], [], [], [], []]
    scatter_y = [[], [], [], [], [], []]
    colors = ['r', 'g', 'b', 'm', 'c', 'k']
    labels = ['-1@train', '0@train', '1@train', '-1@test', '0@test', '1@test']

    for i, y in enumerate(train_label):
        scatter_x[y+1].append(_data[i][0])
        scatter_y[y+1].append(_data[i][1])

    for i, y in enumerate(test_label):
        scatter_x[y+4].append(_data[i+len(train_label)][0])
        scatter_y[y+4].append(_data[i+len(train_label)][1])

    for i in range(6):
        plt.scatter(scatter_x[i], scatter_y[i], c=colors[i], label=labels[i])

    plt.legend()
    plt.savefig('visualization.jpg')
    plt.show()


def one_versus_all(train_data, train_label, test_data, test_label, t=0, c=1.):
    m = svm_train(
        svm_problem(train_label, train_data),
        svm_parameter('-b 1 -t %i -q -c %f' % (t, c))
    )

    """
    obj is the optimal objective value of the dual SVM problem. 
    rho is the bias term in the decision function sgn(w^Tx - rho). 
    nSV and nBSV are number of support vectors and bounded support vectors (i.e., alpha_i = C). 
    nu-svm is a somewhat equivalent form of C-SVM where C is replaced by nu. 
    nu simply shows the corresponding parameter.
    """

    # p_label, p_acc, p_val = svm_predict(train_label, train_data, m)
    p_label, p_acc, p_val = svm_predict(test_label, test_data, m, '-b 1 -q')


def ovr_test(models, test_data, test_label):
    vals = []
    for i, model in enumerate(models):
        labels = [1 if l == i-1 else 0 for l in test_label]
        p_label, p_acc, p_val = svm_predict(labels, test_data, model, '-b 1 -q')
        ind = 1 if p_val[0][int(p_label[0])] > p_val[0][1-int(p_label[0])] else 0
        vals.append(np.array(p_val)[:, ind])
    predict = np.argmax(vals, axis=0) - 1
    acc = np.count_nonzero(predict==test_label) / len(test_label)
    return acc


def one_versus_rest(train_data, train_label, test_data, test_label, t, c, g):
    models = []
    params = ('-b 1 -t %i -q -c %f -g %f' % (t, c, g))
    models.append(svm_train(
        svm_problem([1 if l == -1 else 0 for l in train_label], train_data),
        svm_parameter(params),
    ))
    models.append(svm_train(
        svm_problem([1 if l == 0 else 0 for l in train_label], train_data),
        svm_parameter(params),
    ))
    models.append(svm_train(
        svm_problem([1 if l == 1 else 0 for l in train_label], train_data),
        svm_parameter(params),
    ))
    # ovr_test(models, train_data, train_label)
    acc = ovr_test(models, test_data, test_label)
    print('Accuracy = %.5f, t = %i, c = %.6f, g = %.6f' % (acc, t, c, g))
    return acc


def ovo_test(models, test_data, test_label):
    labels = []
    predict = []
    for model in models:
        p_label, p_acc, p_val = svm_predict(test_label, test_data, model, '-b 1 -q')
        labels.append(p_label)
    for i in range(len(test_label)):
        if labels[0][i] == -1 and labels[1][i] == -1:
            predict.append(-1)
        elif labels[0][i] == 0 and labels[2][i] == 0:
            predict.append(0)
        elif labels[1][i] == 1 and labels[2][i] == 1:
            predict.append(1)
        else:
            predict.append(-1)
    acc = np.count_nonzero(predict==test_label) / len(test_label)
    return acc


def one_versus_one(train_data, train_label, test_data, test_label, t, c, g):
    X = [
        train_data[train_label==-1, :],
        train_data[train_label==0, :],
        train_data[train_label==1, :],
    ]
    y = [
        train_label[train_label==-1],
        train_label[train_label==0],
        train_label[train_label==1],
    ]
    params = ('-b 1 -t %i -q -c %f -g %f' % (t, c, g))

    models = []
    models.append(svm_train(
        svm_problem(np.concatenate([y[0], y[1]]), np.concatenate([X[0], X[1]])),
        svm_parameter(params),
    ))
    models.append(svm_train(
        svm_problem(np.concatenate([y[0], y[2]]), np.concatenate([X[0], X[2]])),
        svm_parameter(params),
    ))
    models.append(svm_train(
        svm_problem(np.concatenate([y[1], y[2]]), np.concatenate([X[1], X[2]])),
        svm_parameter(params),
    ))

    # ovo_test(models, train_data, train_label)
    acc = ovo_test(models, test_data, test_label)
    print('Accuracy = %.5f, t = %i, c = %.6f, g = %.6f' % (acc, t, c, g))
    return acc


def main():
    train_data = sio.loadmat('hw3_data/train_data.mat')['train_data']
    train_label = np.squeeze(sio.loadmat('hw3_data/train_label.mat')['train_label'])
    test_data = sio.loadmat('hw3_data/test_data.mat')['test_data']
    test_label = np.squeeze(sio.loadmat('hw3_data/test_label.mat')['test_label'])

    train_data, test_data = scale(train_data, test_data)
    visualize(train_data, train_label, test_data, test_label)

    res = [[np.zeros((21, 21)), np.zeros((21, 21))] for _ in range(2)]
    for k in range(2):
        t = [0, 2][k]
        print('\ntuning ovr parameter for %s kernel' % ['linear', 'rbf'][k])
        ovr_best, ovr_best_c = 0, 0

        for i in range(-10, 11):
            for j in range(-10, 11):
                c = 2**i
                g = 2**j
                acc = one_versus_rest(train_data, train_label, test_data, test_label, t, c, g)
                if acc > ovr_best:
                    ovr_best = acc
                    ovr_best_c = c
                res[k][0][i+10, j+10] = acc
        res[k][0].tofile('ovr_%s.csv' % ['linear', 'rbf'][k], sep=',')

        print('\ntuning ovo parameter for %s kernel' % ['linear', 'rbf'][k])
        ovo_best, ovo_best_c = 0, 0
        for i in range(-10, 11):
            for j in range(-10, 11):
                c = 2**i
                g = 2**j
                acc = one_versus_one(train_data, train_label, test_data, test_label, t, c, g)
                if acc > ovo_best:
                    ovo_best = acc
                    ovo_best_c = c
                res[k][1][i+10, j+10] = acc
        res[k][1].tofile('ovo_%s.csv' % ['linear', 'rbf'][k], sep=',')

        print('ovr best accuracy: %f, c: %f' % (ovr_best, ovr_best_c))
        print('ovo best accuracy: %f, c: %f' % (ovo_best, ovo_best_c))
        print('\n///////////////////////////////////////////////////')

    print('End.')


if __name__ == '__main__':
    main()
