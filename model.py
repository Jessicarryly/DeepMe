import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from utils import weight_variable, bias_variable, conv2d, max_pool_2x2, variable_summaries

"""
The cnn architecture, include
- simple linear regression
- simple CNN network
- Alexnet
"""
class Model(object):
    """
    Provide the model adapt to the input X and Y
    Return logits, loss, keep_prob for tensorflow sess
    name to store the model and draw summary graphs
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def model(self, X, Y):
        pass
    
class LinearRegression(Model):
    """
    Provide the simple Linear regression model, only has 1 linear function
    """

    def model(self, X, Y):
        feature = int(np.prod(X.get_shape()[1:]))
        classes = int(np.prod(Y.get_shape()[1:]))
        keep_prob = tf.placeholder(tf.float32)

         # cnn layer
        W = weight_variable([feature, classes])
        b = bias_variable([classes])

        # loss
        logits = tf.matmul(X, W) + b
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y, name='loss')
        loss = tf.reduce_mean(entropy)
        variable_summaries(loss, 'loss')

        return logits, loss, keep_prob, "linear"

class SimpleCNN(Model):
    """
    Provide a simple CNN model with architecture:
    [affine - relu -pool] - [affine - relu] - 2*[fc] - [softmax]
    """

    def model(self, X, Y):
        feature = int(np.prod(X.get_shape()[1:]))
        classes = int(np.prod(Y.get_shape()[1:]))
        x_image = tf.reshape(X, [-1, feature, 1, 1])

        # 1st conv layer
        with tf.name_scope('conv1'):
            W = weight_variable([5, 1, 1, 32])
            b = bias_variable([32])
            h = tf.nn.relu(conv2d(x_image, W) + b)
            conv1 = max_pool_2x2(h)

        # 2nd conv layer
        with tf.name_scope('conv2'):
            W = weight_variable([5, 1, 32, 64])
            b = bias_variable([64])
            conv2 = tf.nn.relu(conv2d(conv1, W) + b)

        keep_prob = tf.placeholder(tf.float32)

        # 1st fc layer
        with tf.name_scope('fc1'):
            shape = int(np.prod(conv2.get_shape()[1:]))
            W = weight_variable([shape, 1024])
            b = bias_variable([1024])
            conv2_flat = tf.reshape(conv2, [-1, shape])
            h = tf.nn.relu(tf.matmul(conv2_flat, W) + b)
            fc1 = tf.nn.dropout(h, keep_prob)

        # 2nd fc layer
        with tf.name_scope('fc2'):
            W = weight_variable([1024, classes])
            b = bias_variable([classes])
            logits = tf.matmul(fc1, W) + b
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y, name='loss')
            loss = tf.reduce_mean(entropy)
            variable_summaries(loss, 'loss')
        
        return logits, loss, keep_prob, 'cnn'

    
class AlexNet(Model):
    """
    Provide AlextNet model
    5*[conv-relu] - 3*[fc] - [soltmax]
    """

    def model(self, X, Y):
        feature = int(np.prod(X.get_shape()[1:]))
        classes = int(np.prod(Y.get_shape()[1:]))
        x_image = tf.reshape(X, [-1, feature, 1, 1])

        # 1st conv layer
        with tf.name_scope('conv1') as scope:
            w = weight_variable([5, 1, 1, 32])
            b = bias_variable([32])
            h = tf.nn.relu(conv2d(x_image, w) + b)
            conv1 = max_pool_2x2(h)
            # print "conv1 shape: ", h.get_shape()
            # print "pool1 shape: ", conv1.get_shape()

        # 2nd conv layer
        with tf.name_scope('conv2') as scope:
            w = weight_variable([5, 1, 32, 64])
            b = bias_variable([64])
            h = tf.nn.relu(conv2d(conv1, w) + b)
            conv2 = max_pool_2x2(h)
            # print "conv2 shape: ", h.get_shape()
            # print "pool2 shape: ", conv2.get_shape()
        
        # 3rd conv layer
        with tf.name_scope('conv3') as scope:
            w = weight_variable([5, 1, 64, 64])
            b = bias_variable([64])
            conv3 = tf.nn.relu(conv2d(conv2, w) + b)
            # print "conv3 shape: ", conv3.get_shape()

        
        # 4th conv layer
        with tf.name_scope('conv4') as scope:
            w = weight_variable([5, 1, 64, 64])
            b = bias_variable([64])
            conv4 = tf.nn.relu(conv2d(conv3, w) + b)
            # print "conv4 shape: ", conv4.get_shape()

        # 5th conv layer
        with tf.name_scope('conv5') as scope:
            w = weight_variable([5, 1, 64, 64])
            b = bias_variable([64])
            h = tf.nn.relu(conv2d(conv4, w) + b)
            conv5 = max_pool_2x2(h)
            # print "conv5 shape: ", h.get_shape()
            # print "pool5 shape: ", conv5.get_shape()

        # dropout
        keep_prob = tf.placeholder(tf.float32)

        # 1st fc layer
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(conv5.get_shape()[1:]))
            print('shape: ', shape)
            conv5_flat = tf.reshape(conv5, [-1, shape])
            w = weight_variable([shape, 1024])
            b = bias_variable([1024])
            
            h = tf.nn.relu(tf.matmul(conv5_flat, w) + b)
            fc1 = tf.nn.dropout(h, keep_prob)
            # print "fc1 shape: ", fc1.get_shape()


        # 2nd fc layer
        with tf.name_scope('fc2') as scope:
            w = weight_variable([1024, 512])
            b = bias_variable([512])
            h = tf.nn.relu(tf.matmul(fc1, w) + b)
            fc2 = tf.nn.dropout(h, keep_prob)
            # print "fc2 shape: ", fc2.get_shape()

        # 3rd fc layer
        with tf.name_scope('fc3') as scope:
            w = weight_variable([512, classes])
            b = bias_variable([classes])
            logits = tf.matmul(fc2, w) + b
            # print "logits shape: ", logits.get_shape()
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits, name='loss')
            loss = tf.reduce_mean(entropy)
            variable_summaries(loss, 'loss')

        return logits, loss, keep_prob, "alex"

