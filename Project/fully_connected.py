import sys
import scipy.io as sio
import numpy as np
import tensorflow as tf
import logging

LEARNING_RATE = 5e-5
MAX_EPOCHES = int(1e6)


class FullyConnected(object):
    def __init__(self, name, logger, lr, lr_decay, n_classes, max_epoches, train_data, train_label,
                 test_data=None, test_label=None, seed=0):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s\t%(message)s')

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

        file_handler = logging.FileHandler('logs/%s.log' % name)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

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
        self.lr_decay = lr_decay
        self.max_epoches = max_epoches
        self.sess = tf.Session()

        self.LR = tf.placeholder(tf.float32, [], 'learning_rate')
        self.X = tf.placeholder(tf.float32, [None, self.feature_dim], 'obs')
        self.Y = tf.placeholder(tf.int32, [None], 'label')

        with tf.variable_scope('model_%s' % self.name):
            self.logits_ = tf.layers.dense(
                self.X,
                self.n_classes,
                kernel_initializer=tf.random_normal_initializer
            )

        self.loss_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits_,
            labels=self.Y
        ))

        self.opt_ = tf.train.GradientDescentOptimizer(self.LR).minimize(self.loss_)
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        for ep in range(self.max_epoches):
            lr = self.lr * (1 - ep/self.max_epoches) if self.lr_decay else self.lr
            loss, _ = self.sess.run([self.loss_, self.opt_], feed_dict={
                self.X: self.train_data,
                self.Y: self.train_label,
                self.LR: lr
            })
            if ep % 10 == 0:
                if self.to_test:
                    self.logger.info('ep:%i\t loss:%f\t acc:%f' % (ep, loss, self.test()))
                else:
                    self.logger.info('ep:%i\tloss:%f' % (ep, loss))
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
    
    model = FullyConnected(
        name='fully_connected',
        logger=logging.getLogger('fully_connected'),
        lr=LEARNING_RATE,
        lr_decay=False,
        n_classes=4,
        max_epoches=MAX_EPOCHES,
        train_data=train_de,
        train_label=train_label_eeg,
        test_data=test_de,
        test_label=test_label_eeg
    )
    model.train()


if __name__ == '__main__':
    main()
