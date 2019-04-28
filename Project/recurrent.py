import os
import scipy.io as sio
import numpy as np
import tensorflow as tf
from utils import getLogger, timed, parse_arg

args, abstract = parse_arg()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


class RNN(object):
    def __init__(self, folder, name, hidsize, ac_fn, lr, lr_decay, n_classes, train_data, train_label,
                 test_data=None, test_label=None, seed=0, validate=False):
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
        self.hidsize = list(map(int, hidsize.split(',')))
        self.feature_dim = self.train_data.shape[-1]
        self.n_classes = n_classes
        self.lr = lr
        self.lr_decay = lr_decay

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

        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        with self.graph.as_default():
            self.LR = tf.placeholder(tf.float32, [], 'learning_rate')
            self.X = tf.placeholder(tf.float32, [None, self.feature_dim], 'obs')
            self.Y = tf.placeholder(tf.int32, [None], 'label')
            self.train_d = tf.reshape(self.X, [-1, 5, self.feature_dim//5])

            # lstm_cells = [tf.nn.rnn_cell.LSTMCell(dim) for dim in self.hidsize]
            # cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
            # initial_state = cell.zero_state(tf.shape(self.train_d)[0], tf.float32)
            # outputs, final_state = tf.nn.dynamic_rnn(cell, self.train_d, initial_state=initial_state)

            inputs = tf.unstack(self.train_d, axis=1)
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidsize[0], activation=self.ac_fn)
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidsize[0], activation=self.ac_fn)
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32)

            self.logits = tf.layers.dense(
                tf.concat(outputs, axis=-1), self.n_classes,
                kernel_initializer=tf.random_normal_initializer
            )
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.Y
            ))
            self.soft = tf.nn.softmax(self.logits)
            self.opt = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

    def train(self, epoches):
        with timed('training %i epoches' % epoches, self.logger):
            for ep in range(epoches):
                lr = self.lr * (1 - ep / epoches) if self.lr_decay else self.lr
                loss, _ = self.sess.run([self.loss, self.opt], feed_dict={
                    self.X: self.train_data,
                    self.Y: self.train_label,
                    self.LR: lr
                })
                if ep % 10 == 0:
                    if self.validate:
                        self.logger.info('ep:%i\t loss:%f\t acc:%f' % (ep, loss, self.val()))
                    else:
                        self.logger.info('ep:%i\tloss:%f' % (ep, loss))
        self.save()

    def test(self):
        with timed('testing', self.logger):
            logits = self.sess.run(self.logits, feed_dict={self.X: self.test_data})
        labels = np.argmax(logits, axis=-1)
        return labels

    def val(self):
        logits = self.sess.run(self.logits, feed_dict={self.X: self.test_data})
        labels = np.argmax(logits, axis=-1)
        return np.count_nonzero(labels == self.test_label) / self.test_label.size

    def classify(self):
        with timed('test', self.logger):
            logits = self.sess.run(self.soft, feed_dict={self.X: self.test_data})[:, 1]  # p(y=1)
        return logits

    def restore(self):
        if os.path.exists(self.ckpt):
            self.saver.restore(self.sess, self.ckpt)

    def save(self):
        self.saver.save(self.sess, self.ckpt)


def main():
    data = sio.loadmat(os.path.join('data', 'data.mat'))

    train_de = data['train_de']
    train_label_eeg = data['train_label_eeg']
    test_de = data['test_de']
    test_label_eeg = data['test_label_eeg']

    folder = os.path.join('logs', 'rnn_{}'.format(abstract))
    if not os.path.exists(folder):
        os.makedirs(folder)

    model = RNN(
        folder=folder,
        name='recurrent',
        hidsize=args.hidsize,
        ac_fn=args.ac_fn,
        lr=args.lr,
        lr_decay=args.lr_decay,
        n_classes=4,
        train_data=train_de,
        train_label=train_label_eeg,
        test_data=test_de,
        test_label=test_label_eeg,
        validate=True
    )
    model.train(args.max_epoches)


if __name__ == '__main__':
    main()
