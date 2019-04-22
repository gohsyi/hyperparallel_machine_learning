import os
import matplotlib.pyplot as plt


for root, dirs, files in os.walk('logs'):
    for f in files:
        if f[0] != '.' and f.split('.')[-1] == 'log':
            print('processing %s' % f)
            p = os.path.join(root, f)
            loss = []
            acc = []
            for line in open(p):
                line = line.split()
                for x in line:
                    x = x.split(':')
                    if len(x) == 1:
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
                plt.savefig(str(p.split('.')[0]) + '_loss.jpg')
                plt.cla()

            if len(acc) > 0:
                plt.plot(acc)
                plt.title('acc')
                plt.savefig(str(p.split('.')[0]) + '_acc.jpg')
                plt.cla()
