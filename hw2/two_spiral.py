import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from contextlib import contextmanager

np.random.seed(1)
tf.random.set_random_seed(1)

@contextmanager
def timed(msg):
    print(msg)
    tstart = time.time()
    yield
    print('done in %.3f seconds' % (time.time() - tstart))


class MLQP(object):
    def __init__(self, training_set, test_set):
        self.hdim = 10   # hidden layer dimension
        self.max_epoches = int(1e7)
        self.loss_threshold = 1e-3
        self.training_set = training_set
        self.test_set = test_set
        self.sess = tf.Session()
        self.o_, self.loss_, self.opt_ = self.setup_network()

    def setup_network(self):
        self.X = tf.placeholder(tf.float32, [None, 2], 'input')
        self.Y = tf.placeholder(tf.float32, [None, 1], 'ground_truth')
        self.LR = tf.placeholder(tf.float32, [], 'learning_rate')

        with tf.variable_scope('network', initializer=tf.random_normal_initializer):
            u1_ = tf.get_variable('u1', [2, self.hdim])
            v1_ = tf.get_variable('v1', [2, self.hdim])
            u2_ = tf.get_variable('u2', [self.hdim, 1])
            v2_ = tf.get_variable('v2', [self.hdim, 1])
            b1_ = tf.get_variable('b1', [self.hdim])
            b2_ = tf.get_variable('b2', [1])
            x1_ = tf.nn.sigmoid(tf.matmul(tf.square(self.X), u1_) + tf.matmul(self.X, v1_) + b1_)
            x2_ = tf.nn.sigmoid(tf.matmul(tf.square(x1_), u2_) + tf.matmul(x1_, v2_) + b2_)

        e_ = tf.reduce_mean(tf.square(self.Y - x2_) / 2)
        train_opt = tf.train.GradientDescentOptimizer(self.LR).minimize(e_)
        return x2_, e_, train_opt

    def learn(self, lr):
        mean_loss = []
        for x in self.training_set:
            loss, _ = self.sess.run([self.loss_, self.opt_], feed_dict={
                self.X: [x[:-1]],
                self.Y: [[0.1 if x[-1] < 0.5 else 0.9]],  # a small trick to accelerate learning
                self.LR: lr,
            })
            mean_loss.append(loss)
        return float(np.mean(mean_loss))

    def train(self, lr):
        self.sess.run(tf.global_variables_initializer())
        losses = []
        for ep in range(self.max_epoches):
            loss = self.learn(lr)
            losses.append(loss)
            if ep % 100 == 0:
                loss = float(np.mean(losses))
                losses = []
                print('ep %i, loss %.5f' % (ep, loss))
                if loss < self.loss_threshold:
                    break

    def test(self, lr):
        X = self.test_set[:, :-1]
        y = self.sess.run(self.o_, feed_dict={self.X: X})
        for i in range(y.shape[0]):
            if y[i][0] > 0.5:
                plt.scatter(X[i][0], X[i][1], c='red')
            else:
                plt.scatter(X[i][0], X[i][1], c='green')
        plt.savefig('lr%f_test.jpg' % lr)
        plt.show()

        X = []
        for i in np.linspace(-3.5, 3.5, 100):
            for j in np.linspace(-3.5, 3.5, 100):
                X.append([i, j])
        y = self.sess.run(self.o_, feed_dict={self.X: X})
        for i in range(y.shape[0]):
            if y[i][0] > 0.5:
                plt.scatter(X[i][0], X[i][1], c='black')
            else:
                plt.scatter(X[i][0], X[i][1], c='white')
        plt.savefig('lr%f_plot.jpg' % lr)
        plt.show()


if __name__ == '__main__':
    mlqp = MLQP(np.loadtxt('hw2_data/two_spiral_train.txt'),
                np.loadtxt('hw2_data/two_spiral_test.txt'))

    for lr in [1e-1, 1e-2, 1e-3]:
        with timed('training with lr = %f' % lr):
            mlqp.train(lr)
        mlqp.test(lr)
