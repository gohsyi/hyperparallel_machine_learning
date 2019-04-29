import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import argparse
import scipy.io as sio
from sklearn.metrics import confusion_matrix, f1_score


parser = argparse.ArgumentParser()
parser.add_argument('-smooth', type=float, default=0)
args = parser.parse_args()

y_true = sio.loadmat(os.path.join('data', 'data.mat'))['test_label_eeg']

for root, dirs, files in os.walk('logs'):
    for f in files:
        if f[0] != '.' and f.split('.')[-1] == 'csv':  # process .csv
            p = os.path.join(root, f)
            print('processing %s' % p)
            y_pred = np.loadtxt(p)
            cm = confusion_matrix(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, labels=range(4), average=None)

            with open(os.path.join(root, f.split('.')[0] + '.txt'), 'w') as f:
                f.write('& neutral & sad & fear & happy \\\\\\hline\n')
                for i in range(4):
                    f.write(['neutral', 'sad', 'fear', 'happy'][i])
                    for j in range(4):
                        f.write(' & %i' % int(cm[i][j]))
                    f.write(' \\\\\\hline\n')

                f.write('\n')

                f.write('& neutral & sad & fear & happy \\\\\\hline\nF1')
                for i in range(4):
                    f.write(' & %.2f' % f1[i])
                f.write(' \\\\\\hline')

        elif f[0] != '.' and f.split('.')[-1] == 'log':  # process .log
            p = os.path.join(root, f)
            print('processing %s' % p)
            loss = []
            acc = []
            for line in open(p):
                line = line.split()
                for x in line:
                    x = x.split(':')
                    if x[0] == 'ep' and x[1] == '0':
                        loss = []
                        acc = []
                    if x[0] == 'loss':
                        loss.append(float(x[1]))
                    if x[0] == 'acc':
                        acc.append(float(x[1]))

            if len(loss) > 0:
                plt.plot(loss)
                plt.title('loss')
                plt.savefig('.'.join(p.split('.')[:-1]) + '_loss.jpg')
                plt.cla()

            if len(acc) > 0:
                for i in range(1, len(acc)):
                    acc[i] = acc[i-1] * args.smooth + acc[i] * (1-args.smooth)
                plt.plot(acc)
                plt.title('acc')
                plt.savefig('.'.join(p.split('.')[:-1]) + '_acc.jpg')
                plt.cla()
