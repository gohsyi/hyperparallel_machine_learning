import random
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use("Solarize_Light2")

np.random.seed(0)
tf.random.set_random_seed(0)


class Model(object):
    def __init__(self):
        self.n1 = 500
        self.n2 = 500
        self.training_portion = 0.7
        self.max_epoches = 20000
        self.R = 5  # data range
        self.sess = tf.Session()
        self.data_generating()

    def data_generating(self):
        k, b = np.random.uniform(1, 2), np.random.uniform(-self.R/5, self.R/5)
        n1, n2 = 0, 0
        dataset, training_set, test_set = [], [], []

        while len(dataset) < (self.n1 + self.n2):
            x, y = np.random.uniform(-self.R, self.R), np.random.uniform(-self.R, self.R)
            if y < k * x + b and n1 < self.n1:
                dataset.append((x, y, -1))
                plt.plot(x, y, 'ro')
            elif y > k * x + b and n2 < self.n2:
                dataset.append((x, y, 1))
                plt.plot(x, y, 'go')

        X = np.arange(-self.R, self.R, 0.1)
        Y = k * X + b
        plt.plot(X, Y, 'b', label='y = %.1fx + %.1f' % (k, b))
        plt.ylabel('y')
        plt.xlabel('x')
        plt.title('generated data')
        plt.legend()
        plt.savefig('generated.jpg')
        plt.show()

        random.shuffle(dataset)
        self.training_set = np.array(dataset[:int((self.n1 + self.n2) * self.training_portion)])
        self.test_set = np.array(dataset[int((self.n1 + self.n2) * self.training_portion):])

    def setup_network(self):
        self.X = tf.placeholder(tf.float32, [None, 2], 'input')
        self.Y = tf.placeholder(tf.float32, [None, 1], 'label')
        self.LR = tf.placeholder(tf.float32, [], 'lr')

        w_ = tf.get_variable('w', [2, 1], dtype=tf.float32)
        b_ = tf.get_variable('b', [1], dtype=tf.float32)

        self.o_ = tf.matmul(self.X, w_) + b_
        self.loss_ = tf.reduce_mean(tf.squared_difference(self.o_, self.Y))
        self.opt_ = tf.train.GradientDescentOptimizer(self.LR).minimize(self.loss_)

        self.sess.run(tf.global_variables_initializer())

    def train(self, lr):
        acc = []
        with tf.variable_scope('lr%f' % lr):
            self.setup_network()

        for ep in range(self.max_epoches):
            _, loss = self.sess.run([self.opt_, self.loss_], feed_dict={
                self.LR: lr,
                self.X: self.training_set[:, :-1],
                self.Y: self.training_set[:, -1:],
            })

            # test
            o = self.sess.run(self.o_, feed_dict={self.X: self.test_set[:, :-1],})
            acc.append(np.count_nonzero((o>0) == (self.test_set[:, -1:]==1)) / o.size)

            if ep % (self.max_epoches // 20) == 0:
                print('ep %i, loss: %.3f, acc: %.3f' % (ep, loss, acc[-1]))

        return acc


if __name__ == '__main__':
    model = Model()
    plt.figure()
    for i, lr in enumerate([1e-1, 1e-3, 1e-4]):
        plt.plot(model.train(lr), c=['r', 'g', 'b'][i], label='lr = %f' % lr)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train')
    plt.legend()
    plt.savefig('result.jpg')
    plt.show()
