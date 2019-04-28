import os
import scipy.io as sio
import numpy as np
import tensorflow as tf
from utils import getLogger, timed, parse_arg

args, abstract = parse_arg()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


class RNN(object):
    def __init__(self, folder, name, hidsize, batchsize, ac_fn, max_epoches, lr, lr_decay, n_classes, train_data, train_label,
                 test_data=None, test_label=None, seed=0, validate=False):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        assert validate is False or test_data is not None

        self.folder = folder
        self.name = name
        self.ckpt = os.path.join(self.folder, '{}.ckpt'.format(self.name))
        self.logger = getLogger(folder, name)
        self.validate = validate
        self.train_data = np.array(train_data, np.float32)
        self.train_label = np.squeeze(np.array(train_label, np.int32))
        self.test_data = np.array(test_data, np.float32) if test_data is not None else None
        self.test_label = np.squeeze(np.array(test_label, np.int32)) if test_label is not None else None
        self.batchsize = batchsize
        self.hidsize = list(map(int, hidsize.split(',')))
        self.feature_dim = self.train_data.shape[-1]
        self.n_classes = n_classes
        self.max_epoches = max_epoches
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
            self.train_data = np.reshape(self.train_data, [-1, 5, self.feature_dim // 5])
            self.test_data = np.reshape(self.test_data, [-1, 5, self.feature_dim // 5])
            self.batch = tf.data.Dataset.from_tensor_slices((self.train_data, self.train_label))
            self.batch = self.batch.shuffle(self.train_data.shape[0]).batch(self.batchsize).repeat()
            self.train_d, self.train_l = self.batch.make_one_shot_iterator().get_next()

            # lstm_cells = [tf.nn.rnn_cell.LSTMCell(dim) for dim in self.hidsize]
            # cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
            # initial_state = cell.zero_state(tf.shape(self.train_d)[0], tf.float32)
            # outputs, final_state = tf.nn.dynamic_rnn(cell, self.train_d, initial_state=initial_state)

            inputs = tf.unstack(self.train_d, axis=1)
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidsize[0], activation=self.ac_fn)
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidsize[0], activation=self.ac_fn)
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32)

            self.hidden = [tf.concat(outputs, axis=-1)]

            for hdim in self.hidsize[1:]:
                self.hidden.append(tf.layers.dense(
                    self.hidden[-1], hdim,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
                ))

            self.logits = tf.layers.dense(
                self.hidden[-1], self.n_classes,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=tf.stop_gradient(tf.one_hot(self.train_l, self.n_classes))
            ))
            self.sigmoid = tf.nn.sigmoid(self.logits)
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

    def train(self):
        with timed('training %i epoches' % self.max_epoches, self.logger):
            for ep in range(self.max_epoches):
                loss, _ = self.sess.run([self.loss, self.opt])
                if ep % 10 == 0:
                    if self.validate:
                        self.logger.info('ep:%i\t loss:%f\t acc:%f' % (ep, loss, self.val()))
                    else:
                        self.logger.info('ep:%i\tloss:%f' % (ep, loss))
        self.save()

    def test(self):
        with timed('testing', self.logger):
            logits = self.sess.run(self.logits, feed_dict={self.train_d: self.test_data})
        labels = np.argmax(logits, axis=-1)
        return labels

    def val(self):
        logits = self.sess.run(self.logits, feed_dict={self.train_d: self.test_data})
        labels = np.argmax(logits, axis=-1)
        return np.count_nonzero(labels == self.test_label) / self.test_label.size

    def classify(self):
        with timed('test', self.logger):
            logits = self.sess.run(self.sigmoid, feed_dict={self.train_d: self.test_data})[:, 1]  # p(y=1)
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
        batchsize=args.batchsize,
        ac_fn=args.ac_fn,
        max_epoches=args.max_epoches,
        lr=args.lr,
        lr_decay=args.lr_decay,
        n_classes=4,
        train_data=train_de,
        train_label=train_label_eeg,
        test_data=test_de,
        test_label=test_label_eeg,
        validate=True
    )
    model.train()


if __name__ == '__main__':
    main()
