"""
The util class for common use function in the project
"""
from scipy.io import loadmat
from scipy.fftpack import fft
from scipy.signal import butter, firwin, lfilter
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gnuplotlib as gp
from os.path import join
from csv import reader

def weight_variable(shape, name='w'):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name='b'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name='pool'):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name=name)

def load_data(path, csvfile, percent=100, all_feature=False, length=2048, ids={}):
    """
    Load all data from the current path, with label save in .csv file
    percent: percentage of data to use
    all_feature: all data will be truncated wrt the max len data
    length: provide the len of the data for trimming, or truncating, be carefull with this
    id: map the label from string to index

    Return the ndarray of data and how many data of each kind in the data set
    """
    X = []
    Y = []
    with open(join(path, csvfile), 'rb') as file:
        ecgreader = reader(file, delimiter=',')
        for row in ecgreader:
            val = loadmat(join(path, row[0]))['val']
            label = row[1]
            if all_feature:
                x = np.zeros(len)
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

    # count the number of N, F, O, ~; return the result as a tuple
    count = np.bincount(Y)

    Y = np.eye(len(ids.keys()))[Y] # Y = 1 where label index

    return X, Y, count

def preprocess_data(X, preprocess=True):
    """
    preprocess the data with fft, filter, pankin tomson, wavelet, whatever work
    X; Input signal to be preprocess
    return X after preprocessing
    """
    if preprocess:
        return fir(X)
    else:
        return X

# fir low pass filter with cutoff frequency of 40Hz
def fir(x):
    fs = 300.0
    fc = 40.0
    Wc = fc / (fs / 2)
    N = 41
    b = firwin(N, Wc)
    return lfilter(b, 1, x)

# iir lowpass filter with cutoff frequency of 40Hz
def iir(x):
    fs = 300.0
    fc = 40.0
    Wc = fc / (fs / 2)
    N = 41
    [b, a] = butter(N, Wc)
    return lfilter(b, a, x)

def plot_mat(file, title, path, len=2048):
    """
    Plot the data from .mat file to terminal
    """
    val = loadmat(join(path, file))['val'][:, 0:len]
    gp.plot(val, title=title, xlabel='Time (s)', ylabel='Amplitude (mV)', _with='lines', terminal='dumb 80, 40', unset='grid')

def plot(X, len=1200):
    """
    Plot the data of X to terminal
    """
    gp.plot(X[:, 0:len], xlabel='Time (s)', ylabel='Amplitude (mV)', _with='lines', terminal='dumb 80, 40', unset='grid')
