import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from utils import weight_variable, bias_variable, conv2d, max_pool_2x2

"""
The cnn architecture, include
- simple linear regression
- simple CNN network
- Alexnet
"""
class Model(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def model(self, X, Y):
        pass
    
class LinearRegression(Model):

    def model(self, X, Y):
        feature = int(np.prod(X.get_shape()[1:]))
        classes = int(np.prod(Y.get_shape()[1:]))
        keep_prob = tf.placeholder(tf.float32)

         # cnn layer
        W = tf.Variable(tf.truncated_normal([feature, classes]), name='Weights')
        b = tf.Variable(tf.zeros(classes), name='biases')
        tf.summary.histogram('weight', W)
        tf.summary.histogram('bias', b)

        # loss
        logits = tf.matmul(X, W) + b
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y, name='loss')
        loss = tf.reduce_mean(entropy)
        tf.summary.histogram('loss', loss)

        return logits, loss, keep_prob, "linear"

    
# class AlexNet(Model):

#     def model(self, X, Y):
#         # 1st conv layer
#         with tf.name_scope('conv1') as scope:
#             w = weight_variable([5, 1, 1, 32])
#             b = bias_variable([32])
#             h = tf.nn.relu(conv2d(x_image, w) + b)
#             self.conv1 = max_pool_2x2(h)

#         # 2nd conv layer
#         with tf.name_scope('conv2') as scope:
#             w = weight_variable([5, 1, 32, 64])
#             b = bias_variable([64])
#             h = tf.nn.relu(conv2d(self.conv1, w) + b)
#             self.conv2 = max_pool_2x2(h)
        
#         # 3rd conv layer
#         with tf.name_scope('conv3') as scope:
#             w = weight_variable([5, 1, 64, 64])
#             b = bias_variable([64])
#             self.conv3 = tf.nn.relu(conv2d(self.conv2, w) + b)
        
#         # 4th conv layer
#         with tf.name_scope('conv4') as scope:
#             w = weight_variable([5, 1, 64, 64])
#             b = bias_variable([64])
#             self.conv4 = tf.nn.relu(conv2d(self.conv3, w) + b)

#         # 5th conv layer
#         with tf.name_scope('conv5') as scope:
#             w = weight_variable([5, 1, 64, 64])
#             b = bias_variable([64])
#             h = tf.nn.relu(conv2d(self.conv4, w) + b)
#             self.conv5 = max_pool_2x2(h)

#         # dropout
#         keep_prob = tf.placeholder(tf.float32)

#         # 1st fc layer
#         with tf.name_scope('fc1') as scope:
#             shape = int(np.prod(self.conv5.get_shape()[1:]))
#             conv5_flat = tf.reshape(self.conv5, [-1, shape])
#             # w = weight_variable([self.ecg.nfeatures*64, 1024])
#             w = weight_variable([shape, 1024])
#             b = bias_variable([1024])
            
#             h = tf.nn.relu(tf.matmul(conv5_flat, w) + b)
#             self.fc1 = tf.nn.dropout(h, self.keep_prob)

#         # 2nd fc layer
#         with tf.name_scope('fc2') as scope:
#             w = weight_variable([1024, 512])
#             b = bias_variable([512])
#             h = tf.nn.relu(tf.matmul(self.fc1, w) + b)
#             self.fc2 = tf.nn.dropout(h, self.keep_prob)

#         # 3rd fc layer
#         with tf.name_scope('fc3') as scope:
#             w = weight_variable([512, self.ecg.nclasses])
#             b = bias_variable([self.ecg.nclasses])
#             logits = tf.matmul(self.fc2, w) + b
#             entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.Y, name='loss')
#             loss = tf.reduce_mean(entropy

#             return logits, loss, keep_prob, "alex"
