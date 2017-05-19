"""
The util class for common use function in the project
"""
from scipy.io import loadmat
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gnuplotlib as gp
from os.path import join
from csv import reader

"""
utility function for constructing the cnn model
"""
def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

"""
utility for visualize the data model
"""
def load_data(path, csvfile, percent=100, all_feature=False, length=2714, ids={}):
    X = []
    Y = []
    with open(join(path, csvfile), 'rb') as file:
        ecgreader = reader(file, delimiter=',')
        for row in ecgreader:
            val = loadmat(join(path, row[0]))['val']
            label = row[1]
            if all_feature:
                x = np.zeros(self.maxlen)
                n = val.shape[1]
                val = val.reshape(n)
                x[0:n] = val
                X.append(x)
            else:
                X.append(val[:, 0:length].reshape(length))
            Y.append(ids[label])

    # convert of python list to ndarray
    X = np.asarray(X)
    Y = np.asarray(Y)

    # only take a subset of the data
    if percent < 100:
        N, d = X.shape
        n = N / percent
        idx = np.random.choice(range(N), n, replace=False)
        X = X[idx, :]
        Y = Y[idx]

    Y = np.eye(len(ids.keys()))[Y]

    return X, Y

def preprocess_data(X, Y):
    """
    preprocess the data with fft, filter, pankin tomson, wavelet, whatever work
    """
    return X, Y


def plot_mat(file, title, path, len=2714):
    val = loadmat(join(path, file))['val'][:, 0:len]
    gp.plot(val, title=title, xlabel='Time (s)', ylabel='Amplitude (mV)', _with='lines', terminal='dumb 120, 40', unset='grid')
    # gp.xlabel('Time (s)')
    # gp.ylabel('Amplitude (mV)')
    # gp.title(title)
    # plt.show()
