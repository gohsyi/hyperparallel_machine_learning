import sys
sys.path.append('/usr/local/3rdparty/libsvm/python/')
import scipy.io as sio
import numpy as np
from svmutil import *

train_data = sio.loadmat('hw3_data/train_data.mat')['train_data']
train_label = np.squeeze(sio.loadmat('hw3_data/train_label.mat')['train_label'])
test_data = sio.loadmat('hw3_data/test_data.mat')['test_data']
test_label = np.squeeze(sio.loadmat('hw3_data/test_label.mat')['test_label'])

m = svm_train(
    svm_problem(train_label, train_data),
    svm_parameter('-t 0 -c 4 -b 1')
)
