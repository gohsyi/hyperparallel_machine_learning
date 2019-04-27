import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")


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
                plt.plot(acc)
                plt.title('acc')
                plt.savefig('.'.join(p.split('.')[:-1]) + '_acc.jpg')
                plt.cla()
