import tensorflow as tf
import time
from ecg import ECG
from utils import *

"""
the model that will perform the training on the dataset
"""
class CNN:
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
        [affine - relu - maxpool] - [affine - relu - maxpool] - [fc] - [softmax]
        """

        # Input to network, the number of feature is the power of 2
        self.X = tf.placeholder(tf.float32, [None, self.ecg.nfeatures], name='X_placeholder')
        self.Y = tf.placeholder(tf.float32, [None, self.ecg.nclasses], name='Y_placeholder')
        x_image = tf.reshape(self.X, [-1, self.ecg.nfeatures, 1, 1])

        # 1st conv layer
        with tf.name_scope('conv1') as scope:
            w = weight_variable([5, 1, 1, 32])
            b = bias_variable([32])
            h = tf.nn.relu(conv2d(x_image, w) + b)
            self.conv1 = max_pool_2x2(h)

        # 2nd conv layer
        with tf.name_scope('conv2') as scope:
            w = weight_variable([5, 1, 32, 64])
            b = bias_variable([64])
            h = tf.nn.relu(conv2d(self.conv1, w) + b)
            self.conv2 = max_pool_2x2(h)
        
        # 3rd conv layer
        with tf.name_scope('conv3') as scope:
            w = weight_variable([5, 1, 64, 64])
            b = bias_variable([64])
            self.conv3 = tf.nn.relu(conv2d(self.conv2, w) + b)
        
        # 4th conv layer
        with tf.name_scope('conv4') as scope:
            w = weight_variable([5, 1, 64, 64])
            b = bias_variable([64])
            self.conv4 = tf.nn.relu(conv2d(self.conv3, w) + b)

        # 5th conv layer
        with tf.name_scope('conv5') as scope:
            w = weight_variable([5, 1, 64, 64])
            b = bias_variable([64])
            h = tf.nn.relu(conv2d(self.conv4, w) + b)
            self.conv5 = max_pool_2x2(h)

        # dropout
        self.keep_prob = tf.placeholder(tf.float32)

        # 1st fc layer
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.conv5.get_shape()[1:]))
            conv5_flat = tf.reshape(self.conv5, [-1, shape])
            # w = weight_variable([self.ecg.nfeatures*64, 1024])
            w = weight_variable([shape, 1024])
            b = bias_variable([1024])
            
            h = tf.nn.relu(tf.matmul(conv5_flat, w) + b)
            self.fc1 = tf.nn.dropout(h, self.keep_prob)

        # 2nd fc layer
        with tf.name_scope('fc2') as scope:
            w = weight_variable([1024, 512])
            b = bias_variable([512])
            h = tf.nn.relu(tf.matmul(self.fc1, w) + b)
            self.fc2 = tf.nn.dropout(h, self.keep_prob)

        # 3rd fc layer
        with tf.name_scope('fc3') as scope:
            w = weight_variable([512, self.ecg.nclasses])
            b = bias_variable([self.ecg.nclasses])
            self.logits = tf.matmul(self.fc2, w) + b
            entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.Y, name='loss')
            self.loss = tf.reduce_mean(entropy)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def __init_session(self):
        """
        Init the tensorflow session and path to save model
        """
        self.sess = tf.Session()
        self.save_path = 'tmp/alex.ckpt'
        self.id_to_class_name = {0: 'Normal', 1: 'AF', 2: 'Other', 3: 'Noise'}

    def train(self):
        """
        start training the model using the setting in init
        """

        print 'start training the cnn'

        # setup before train
        start = time.time() # measure training time
        writer = tf.summary.FileWriter('graphs', 
                                        self.sess.graph) # graph to visualize the training better

        self.sess.run(tf.global_variables_initializer()) # init all variables
        batches = self.ecg.ntrains / self.batch_size # get the number of batches for each epoch
        saver = tf.train.Saver()

        # train
        for i in range(self.epochs):
            loss = 0
            for _ in range(batches):
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
        fn = 2.0 * corrects[0] / (total[0] + self.ecg.N)
        fa = 2.0 * corrects[1] / (total[1] + self.ecg.A)
        fo = 2.0 * corrects[2] / (total[2] + self.ecg.O)
        fp = 2.0 * corrects[3] / (total[3] + self.ecg.P)
        f = (fn + fa + fo + fp) / 4.0
        print 'Accuracy in the validation set is {0}'.format(f)
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
            X = loadmat(file)['val'][:, 0:2048]
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
