import tensorflow as tf
import time
from ecg import ECG
from utils import *

"""
the model that will perform the training on the dataset
"""
class VGG:
    def __init__(self, ecg=ECG(), learning_rate=1e-3, epochs=30, batch_size=128, dropout=0.75, develop=False):
        """
         init the convolution neural network for training
         ecg: the data model that will provide data for each batch
         learning_rate: the learning rate to fetch to tensorflow
         epochs: number of iteration time throught the dataset
         batch_size: size of batch for each iteration
         dropout: reduce overfitting
         develop: if the model is being developed, use a smaller set of data and epochs
        """
        print 'setup convolution neural network'
        self.ecg = ecg
        self.learning_rate = learning_rate
        if develop:
            self.epochs = 5
        else:
            self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.__init_layer()
        self.__init_session()


    def __init_layer(self):
        """
        setup all the layer needed, following the architecture
        [affine - relu - maxpool] - [affine - relu] - [fc] - [softmax]
        """

        # Input to network, the number of feature is the power of 2
        self.X = tf.placeholder(tf.float32, [None, self.ecg.nfeatures], name='X_placeholder')
        self.Y = tf.placeholder(tf.float32, [None, self.ecg.nclasses], name='Y_placeholder')
        ecg = tf.reshape(self.X, [-1, self.ecg.nfeatures, 1, 1])

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            W = weight_variable([5, 1, 1, 64])
            b = bias_variable([64])
            self.conv1_1 = tf.nn.relu(conv2d(ecg, W) + b)

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            W = weight_variable([5, 1, 64, 64])
            b = bias_variable([64]);
            self.conv1_2 = tf.nn.relu(conv2d(self.conv1_1, W) + b)

        # pool1
        self.pool1 = max_pool_2x2(self.conv1_2, name='pool1');

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            W = weight_variable([5, 1, 64, 128]);
            b = bias_variable([128]);
            self.conv2_1 = tf.nn.relu(conv2d(self.pool1, W) + b)

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            W = weight_variable([5, 1, 128, 128]);
            b = bias_variable([128]);
            self.conv2_2 = tf.nn.relu(conv2d(self.conv2_1, W) + b)

        # pool2:
        self.pool2 = max_pool_2x2(self.conv2_2, name='pool2');

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            W = weight_variable([5, 1, 128, 256]);
            b = bias_variable([256]);
            self.conv3_1 = tf.nn.relu(conv2d(self.pool2, W) + b)

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            W = weight_variable([5, 1, 256, 256]);
            b = bias_variable([256]);
            self.conv3_2 = tf.nn.relu(conv2d(self.conv3_1, W) + b)

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            W = weight_variable([5, 1, 256, 256]);
            b = bias_variable([256]);
            self.conv3_3 = tf.nn.relu(conv2d(self.conv3_2, W) + b)

        # pool3
        self.pool3 = max_pool_2x2(self.conv3_3, name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            W = weight_variable([5, 1, 256, 512]);
            b = bias_variable([512]);
            self.conv4_1 = tf.nn.relu(conv2d(self.pool3, W) + b)

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            W = weight_variable([5, 1, 512, 512]);
            b = bias_variable([512]);
            self.conv4_2 = tf.nn.relu(conv2d(self.conv4_1, W) + b)

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            W = weight_variable([5, 1, 512, 512]);
            b = bias_variable([512]);
            self.conv4_3 = tf.nn.relu(conv2d(self.conv4_2, W) + b)

        # pool4
        self.pool4 = max_pool_2x2(self.conv4_3, name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            W = weight_variable([5, 1, 512, 512]);
            b = bias_variable([512]);
            self.conv5_1 = tf.nn.relu(conv2d(self.pool4, W) + b)

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            W = weight_variable([5, 1, 512, 512]);
            b = bias_variable([512]);
            self.conv5_2 = tf.nn.relu(conv2d(self.conv5_1, W) + b)

        # conv5_3
        with tf.name_scope('conv4_3') as scope:
            W = weight_variable([5, 1, 512, 512]);
            b = bias_variable([512]);
            self.conv5_3 = tf.nn.relu(conv2d(self.conv5_2, W) + b)

        # pool5
        self.pool5 = max_pool_2x2(self.conv5_3, name='pool5')

        self.keep_prob = tf.placeholder(tf.float32)

        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            W = weight_variable([shape, 1024])
            b = bias_variable([1024])
            out = tf.nn.relu(tf.matmul(pool5_flat, W) + b)
            self.fc1 = tf.nn.dropout(out, self.keep_prob)

        with tf.name_scope('fc2') as scope:
            W = weight_variable([1024, 1024])
            b = bias_variable([1024])
            out = tf.nn.relu(tf.matmul(self.fc1, W) + b)
            self.fc2 = tf.nn.dropout(out, self.keep_prob)

        with tf.name_scope('fc3') as scope:
            W = weight_variable([1024, self.ecg.nclasses])
            b = bias_variable([self.ecg.nclasses])
            self.logits = tf.matmul(self.fc2, W) + b
            entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.Y, name='loss')
            self.loss = tf.reduce_mean(entropy)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def __init_session(self):
        """
        Init the tensorflow session and path to save model
        """
        self.sess = tf.Session()
        # TODO: 'models/fft.ckpt'
        self.save_path = 'tmp/vgg16.ckpt'
        self.id_to_class_name = {0: 'Normal', 1: 'AF', 2: 'Other', 3: 'Noise'}

    def train(self):
        """
        start training the model using the setting in init
        """

        print 'start training the cnn'

        # setup before train
        start = time.time() # measure training time
        writer = tf.summary.FileWriter('graphs', self.sess.graph) # graph to visualize the training better
        self.sess.run(tf.global_variables_initializer()) # init all variables
        batches = self.ecg.ntrains / self.batch_size # get the number of batches for each epoch
        saver = tf.train.Saver()

        # train
        for i in range(self.epochs):
            loss = 0
            for j in range(batches):
                X_batch, Y_batch = self.ecg.get_train_batch(self.batch_size)
                _, loss_batch = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: X_batch, self.Y: Y_batch, self.keep_prob: self.dropout})
                loss += loss_batch
            print 'Average loss {0}: {1}'.format(i, loss/batches)

        # training finished
        print 'Total train time {0}'.format(time.time() - start)
        print 'Optimizer finished'

        # Save the sess
        saver.save(self.sess, self.save_path)
        print("Model saved in file: %s" % self.save_path)

    def test(self, sample_every=30, verbose=True):
        """
        run the test on the whole data set
        sample_every: print the result of the model every x time
        verbose: should print result to terminals
        """
        print 'start testing the cnn'
        start = time.time()

        # restore saved model
        saver = tf.train.Saver()
        saver.restore(self.sess, self.save_path)

        # total prediction in each class {Normal, AF, Other, Noise}
        total = {0: 0, 1: 0, 2: 0, 3: 0}
        # total correct prediction in each class {Normal, AF, Other, Noise}
        corrects = {0: 0, 1: 0, 2: 0, 3: 0}

        # run through every single data in test set
        for i in range(self.ecg.ntests):
            # run single forward pass
            X_test, Y_test = self.ecg.get_test(i)
            print X_test.shape
            # no drop out in testing
            loss, logit = self.sess.run([self.loss, self.logits], feed_dict={self.X: X_test, self.Y: Y_test, self.keep_prob: 1})

            # get the prediction
            print logit
            probs = self.sess.run(tf.nn.softmax(logit))
            pred = self.sess.run(tf.argmax(probs, 1))[0]

            correct = np.argmax(Y_test)

            total[pred] += 1
            if pred == correct:
                corrects[pred] += 1

            if verbose and i % sample_every == 0:
                plot(X_test)
                print 'True label is {0}'.format(self.id_to_class_name[correct])
                print 'The model predicts', self.id_to_class_name[pred]

        # calculate the accuracy, base Scoring part at https://physionet.org/challenge/2017/#preparing
        FN = 2.0 * corrects[0] / (total[0] + self.ecg.N)
        FA = 2.0 * corrects[1] / (total[1] + self.ecg.A)
        FO = 2.0 * corrects[2] / (total[2] + self.ecg.O)
        FP = 2.0 * corrects[3] / (total[3] + self.ecg.P)
        F = (FN + FA + FO + FP) / 4.0
        print 'Accuracy in the validation set is {0}'.format(F)
        print 'Testing time {0}'.format(time.time() - start)

    def predict(self, file):
        """
        Run the test on the specific data
        """

        # ensure correct file format
        if not file.endswith('.mat'):
            print 'Incorrect file format'
            return

        try:
            # load .mat file, trim the data
            X = loadmat(file)['val'][:, 0:2714]
            # restore save model
            saver = tf.train.Saver()
            saver.restore(self.sess, self.save_path)

            # run the test
            logit = self.sess.run(self.logits, feed_dict={self.X: X, self.keep_prob: 1})

            # get the prediction
            probs = self.sess.run(tf.nn.softmax(logit))
            pred = self.sess.run(tf.argmax(probs, 1))[0]

            # visualize data
            plot(X)
            print 'The model predicts', self.id_to_class_name[pred]
        except IOError:
            print 'No such file {0}'.format(file)
