import tensorflow as tf
import time
from ecg import ECG


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



class CNN:
    def __init__(self, ecg, learning_rate=1e-3, epochs=30, batch_size=128, dropout=0.75):
        # init the convolution neural network
        print 'setup convolution neural network'
        self.ecg = ecg
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.__setup_model()
        self.sess = tf.Session()

    def __setup_model(self):
        """
        setup all the layer needed, include:
        2 conv layer [affine - relu - maxpool]
        2 fc layer [affine -

        """

        # Input to network
        self.X = tf.placeholder(tf.float32, [None, self.ecg.nfeatures], name='X_placeholder')
        self.Y = tf.placeholder(tf.float32, [None, self.ecg.nclasses], name='Y_placeholder')
        x_image = tf.reshape(self.X, [-1, self.ecg.nfeatures, 1, 1])
        
        # 1st conv layer
        W_conv1 = weight_variable([5, 1, 1, 32], 'W_conv1')
        b_conv1 = bias_variable([32], 'b_conv1')
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # 2nd conv layer
        W_conv2 = weight_variable([5, 1, 32, 64], 'W_conv2')
        b_conv2 = bias_variable([64], 'b_conv2')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # h_pool2 = max_pool_2x2(h_conv2)

        # 1st fc layer
        W_fc1 = weight_variable([self.ecg.nfeatures*32, 1024], 'W_fc1')
        b_fc1 = bias_variable([1024], 'b_fc1')
        h_conv2_flat = tf.reshape(h_conv2, [-1, self.ecg.nfeatures*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # 2nd fc layer
        W_fc2 = weight_variable([1024, self.ecg.nclasses], 'W_fc2')
        b_fc2 = bias_variable([self.ecg.nclasses], 'b_fc2')

        # loss
        self.logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.Y, name='loss')
        self.loss = tf.reduce_mean(entropy)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self):
        print 'start training the cnn'
        # setup before train
        start = time.time()
        writer = tf.summary.FileWriter('graphs', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        batches = self.ecg.ntrains / self.batch_size
        # train
        for i in range(self.epochs):
            loss = 0
            for j in range(batches):
                X_batch, Y_batch = self.ecg.get_train_batch(self.batch_size)
                _, loss_batch = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: X_batch, self.Y: Y_batch, self.keep_prob: self.dropout})
                loss += loss_batch
            print 'Average loss {0}: {1}'.format(i, loss/batches)
        print 'Total train time {0}'.format(time.time() - start)
        print 'Optimizer finished'

    def test(self):
        print 'start testing the cnn'
        start = time.time()
        total_correct_preds = 0
        batches = self.ecg.ntests/self.batch_size
        for i in range(batches):
            X_batch, Y_batch = self.ecg.get_test_batch(self.batch_size)
            _, loss, logit = self.sess.run([self.optimizer, self.loss, self.logits], feed_dict={self.X:X_batch, self.Y: Y_batch, self.keep_prob:1})
            preds = tf.nn.softmax(logit)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds += self.sess.run(accuracy)
        print 'Accuracy {0}'.format(total_correct_preds/self.ecg.ntests)
        print 'Total test time {0}'.format(time.time() - start)


ecg = ECG('training2017')
model = CNN(ecg)
model.train()
model.test()
