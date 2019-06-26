import os
import scipy.io as sio
import numpy as np
import tensorflow as tf
from utils import getLogger, timed

LEARNING_RATE = 1e-4
MAX_EPOCHES = int(1e5)


class FullyConnected(object):
    def __init__(self, folder, name, hidsize, max_epoches, ac_fn, lr, lr_decay, n_classes, train_data, train_label,
                 test_data=None, test_label=None, use_sigmoid=False, seed=0, validate=False):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        assert validate is False or test_data is not None

        self.folder = folder
        self.name = name
        self.ckpt = os.path.join(self.folder, '{}.ckpt'.format(self.name))
        self.logger = getLogger(folder, name)
        self.validate = validate
        self.train_data = np.array(train_data)
        self.train_label = np.squeeze(train_label)
        self.test_data = np.array(test_data)
        self.test_label = np.squeeze(test_label)
        self.hidsz = list(map(int, hidsize.split(',')))
        self.max_epoches = max_epoches
        self.feature_dim = self.train_data.shape[-1]
        self.n_classes = n_classes
        self.lr = lr
        self.lr_decay = lr_decay
        self.use_sigmoid = use_sigmoid

        if ac_fn == 'tanh':
            self.ac_fn = tf.nn.tanh
        elif ac_fn == 'relu':
            self.ac_fn = tf.nn.relu
        elif ac_fn == 'sigmoid':
            self.ac_fn = tf.nn.sigmoid
        elif ac_fn == 'elu':
            self.ac_fn = tf.nn.elu
        else:
            raise ValueError

        # self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        with tf.variable_scope(self.name):
            self.LR = tf.placeholder(tf.float32, [], 'learning_rate')
            self.X = tf.placeholder(tf.float32, [None, self.feature_dim], 'obs')
            self.Y = tf.placeholder(tf.int32, [None], 'label')

            self.hidden = [self.X]
            for dim in self.hidsz:
                self.hidden.append(tf.layers.dense(
                    self.hidden[-1], dim,
                    activation=self.ac_fn,
                    kernel_initializer=tf.random_normal_initializer
                ))
            self.logits = tf.layers.dense(
                self.hidden[-1], self.n_classes,
                kernel_initializer=tf.random_normal_initializer
            )

            if self.use_sigmoid:
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=tf.one_hot(self.Y, self.n_classes)))
            else:
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.Y))

            self.softmax = tf.nn.softmax(self.logits)
            self.sigmoid = tf.nn.sigmoid(self.logits)
            self.opt = tf.train.GradientDescentOptimizer(self.LR).minimize(self.loss)
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

    def train(self):
        avg_loss = []
        with timed('training %i epoches' % self.max_epoches, self.logger):
            for ep in range(self.max_epoches):
                lr = self.lr * (1 - ep / self.max_epoches) if self.lr_decay else self.lr
                loss, _ = self.sess.run([self.loss, self.opt], feed_dict={
                    self.X: self.train_data,
                    self.Y: self.train_label,
                    self.LR: lr
                })
                avg_loss.append(loss)
                if ep % 10 == 0:
                    if self.validate:
                        self.logger.info('ep:{}\t loss:{}\t acc:{}'.format(ep, np.mean(avg_loss), self.test()))
                    else:
                        self.logger.info('ep:{}\tloss:{}'.format(ep, np.mean(avg_loss)))
                    avg_loss = []
        self.save()

    def test(self):
        logits = self.sess.run(self.logits, feed_dict={self.X: self.test_data})
        labels = np.argmax(logits, axis=-1)
        return np.mean(labels==self.test_label)

    def predict(self):
        with timed('predicting', self.logger):
            if self.use_sigmoid:
                logits = self.sess.run(self.sigmoid, feed_dict={self.X: self.test_data})[:, 1]  # p(y=1)
            else:
                logits = self.sess.run(self.softmax, feed_dict={self.X: self.test_data})[:, 1]  # p(y=1)
        return logits

    def restore(self):
        try:
            self.saver.restore(self.sess, self.ckpt)
            self.logger.info('model restored from {}'.format(self.ckpt))
        except:
            self.logger.info('checkpoint does not exist')

    def save(self):
        self.saver.save(self.sess, self.ckpt)
        self.logger.info('model saved in {}'.format(self.ckpt))


def main():
    data = sio.loadmat(os.path.join('data', 'data.mat'))

    train_de = data['train_de']
    train_label_eeg = data['train_label_eeg']
    test_de = data['test_de']
    test_label_eeg = data['test_label_eeg']

    folder = os.path.join('logs', 'fully_connected')
    if not os.path.exists(folder):
        os.makedirs(folder)

    model = FullyConnected(
        folder=folder,
        name='fully_connected',
        hidsize='128',
        max_epoches=MAX_EPOCHES,
        ac_fn='sigmoid',
        lr=LEARNING_RATE,
        lr_decay=False,
        use_sigmoid=True,
        n_classes=4,
        train_data=train_de,
        train_label=train_label_eeg,
        test_data=test_de,
        test_label=test_label_eeg,
        validate=True
    )
    model.train()
    predicts = model.predict()
    np.savetxt(os.path.join(folder, 'predicts.csv'), np.argmax(predicts, axis=1), delimiter=',')


if __name__ == '__main__':
    main()
