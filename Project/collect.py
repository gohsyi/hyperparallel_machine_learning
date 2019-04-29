import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import argparse
import scipy.io as sio
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('-smooth', type=float, default=0)
args = parser.parse_args()

y_true = sio.loadmat(os.path.join('data', 'data.mat'))['test_label_eeg']

for root, dirs, files in os.walk('logs'):
    for f in files:
        try:
            if f[0] != '.' and f.split('.')[-1] == 'csv':  # process .csv
                print('processing %s' % f)
                p = os.path.join(root, f)
                y_pred = np.loadtxt(p)
                cm = confusion_matrix(y_true, y_pred)
                np.savetxt(os.path.join(root, f.split('.')[0] + '_cm.csv'), cm)

            elif f[0] != '.' and f.split('.')[-1] == 'log':  # process .log
                print('processing %s' % f)
                p = os.path.join(root, f)
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
        except:
            print('error occurs')
