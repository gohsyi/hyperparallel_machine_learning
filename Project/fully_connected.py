import sys
import scipy.io as sio
import numpy as np
import tensorflow as tf


LEARNING_RATE = 1e-4
MAX_EPOCHES = int(1e4)


class FullyConnected(object):
    def __init__(self, name, lr, n_classes, max_epoches, train_data, train_label, test_data=None, test_label=None):
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s\t%(message)s')
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

        self.logger = logger
        self.name = name
        self.to_test = (test_data is not None)
        self.train_data = np.array(train_data)
        self.train_label = np.squeeze(train_label)
        self.test_data = np.array(test_data)
        self.test_label = np.squeeze(test_label)
        self.feature_dim = self.train_data.shape[-1]
        self.n_classes = n_classes
        self.lr = lr
        self.max_epoches = max_epoches
        self.sess = tf.Session()

        self.X = tf.placeholder(tf.float32, [None, self.feature_dim], 'obs')
        self.Y = tf.placeholder(tf.int32, [None], 'label')

        with tf.variable_scope('model_%s' % self.name):
            self.logits_ = tf.layers.dense(
                self.X,
                self.n_classes,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer
            )

        self.loss_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits_,
            labels=self.Y
        ))

        self.opt_ = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss_)
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        for ep in range(self.max_epoches):
            loss, _ = self.sess.run([self.loss_, self.opt_], feed_dict={
                self.X: self.train_data,
                self.Y: self.train_label
            })
            if ep % (self.max_epoches//100) == 0:
                if self.to_test:
                    self.logger.info('ep:%i\tloss:%f\tacc:%f' % (ep, loss, self.test()))
                else:
                    # self.logger.info('ep:%i\tloss:%f' % (ep, loss))
                    print('ep:%i\tloss:%f' % (ep, loss))
        # self.logger.info('model %s finished training' % self.name)
        print('model %s finished training' % self.name)

    def test(self):
        logits = self.sess.run(self.logits_, feed_dict={self.X: self.test_data})
        labels = np.argmax(logits, axis=-1)
        acc = np.count_nonzero(labels==self.test_label) / self.test_label.size
        return acc

    def classify(self, X):
        logits = self.sess.run(self.logits_, feed_dict={self.X: X})
        labels = np.argmax(logits, axis=-1)
        return labels


def main():
    data = sio.loadmat('data.mat')

    train_de = data['train_de']
    train_label_eeg = data['train_label_eeg']
    test_de = data['test_de']
    test_label_eeg = data['test_label_eeg']

    model = FullyConnected('fully_connected', LEARNING_RATE, 4, MAX_EPOCHES, train_de, train_label_eeg, test_de, test_label_eeg)
    model.train()


if __name__ == '__main__':
    main()
