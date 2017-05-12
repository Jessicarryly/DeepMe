import tensorflow as tf
import time
from ecg import ECG


class CNN:
    def __init__(self, ecg, learning_rate=1e-3, epochs=5, batch_size=10, batches=6, dropout=0.75):
        # init the convolution neural network
        print 'setup convolution neural network'
        self.ecg = ecg
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.batches = batches
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
        self.X = tf.placeholder(tf.float32, [None, self.ecg.minlen], name='X_placeholder')
        self.Y = tf.placeholder(tf.float32, [None, self.ecg.nclasses], name='Y_placeholder')

        # cnn layer
        W = tf.Variable(tf.truncated_normal([self.ecg.minlen, self.ecg.nclasses]), name='Weights')
        b = tf.Variable(tf.zeros([self.ecg.nclasses]), name='biases')

        # loss
        self.logits = tf.matmul(self.X, W) + b
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

        # train
        for i in range(self.epochs):
            loss = 0
            for j in range(self.batches):
                X_batch, Y_batch = self.ecg.get_train_batch(self.batch_size)
                _, loss_batch = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: X_batch, self.Y: Y_batch})
                loss += loss_batch
            print 'Average loss {0}: {1}'.format(i, loss/self.batches)
        print 'Total time {0}'.format(time.time() - start)
        print 'Optimizer finished'

    def test(self):
        print 'start testing the cnn'

        total_correct_preds = 0
        batches = self.ecg.ntests/self.batch_size
        for i in range(batches):
            X_batch, Y_batch = self.ecg.get_test_batch(self.batch_size)
            _, loss, logit = self.sess.run([self.optimizer, self.loss, self.logits], feed_dict={self.X:X_batch, self.Y: Y_batch})
            preds = tf.nn.softmax(logit)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds += self.sess.run(accuracy)
        print 'Accuracy {0}'.format(total_correct_preds/self.ecg.ntests)



ecg = ECG(basepath='training2017')
model = CNN(ecg)
model.train()
model.test()
