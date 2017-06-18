import tensorflow as tf
import time
from ecg import ECG
from utils import *
from model import *

"""
the model that will perform the training on the dataset
"""
class Solver:
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
        setup all the layer needed, using the class from model.py
        Can easily switch back and fort between model
        """

        # Input to network, the number of feature is the power of 2
        self.X = tf.placeholder(tf.float32, [None, self.ecg.nfeatures], name='X_placeholder')
        self.Y = tf.placeholder(tf.float32, [None, self.ecg.nclasses], name='Y_placeholder')

        # init layer and model name
        self.logits, self.loss, self.keep_prob, name = AlexNet().model(self.X, self.Y)
        self.name = name

        # optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def __init_session(self):
        """
        Init the tensorflow session and path to save model
        """
        self.sess = tf.Session()
        
        self.id_to_class_name = {0: 'Normal', 1: 'AF', 2: 'Other', 3: 'Noise'}

    def train(self):
        """
        start training the model using the setting in init
        """

        print 'start training the cnn'

        # setup before train
        start = time.time() # measure training time

        # visualize the train model
        merged = tf.summary.merge_all()
        writter = tf.summary.FileWriter('graphs/{0}/train'.format(self.name), self.sess.graph) 

        self.sess.run(tf.global_variables_initializer()) # init all variables
        batches = self.ecg.ntrains / self.batch_size # get the number of batches for each epoch
        saver = tf.train.Saver()

        # train
        for i in range(self.epochs):
            loss = 0
            for j in range(batches):
                X_batch, Y_batch = self.ecg.get_train_batch(self.batch_size)
                _, loss_batch, summary = self.sess.run([self.optimizer, self.loss, merged], feed_dict={self.X: X_batch, self.Y: Y_batch, self.keep_prob: self.dropout})
                loss += loss_batch
                step = i * self.epochs + j
                writter.add_summary(summary, step)
            print 'Average loss {0}: {1}'.format(i, loss/batches)

        # training finished
        print 'Total train time {0}'.format(time.time() - start)
        print 'Optimizer finished'

        # Save the sess
        save_path = "model/{0}.ckpt".format(self.name)
        saver.save(self.sess, save_path)
        print("Model saved in file: %s" % save_path)

    def test(self, sample_every=30, verbose=True):
        """
        run the test on the whole data set
        sample_every: print the result of the model every x time
        verbose: should print result to terminals
        """
        print 'start testing the cnn'
        start = time.time()

        # visualize the test model
        merged = tf.summary.merge_all()
        writter = tf.summary.FileWriter('graphs/{0}/test'.format(self.name), self.sess.graph) 

        # restore saved model
        saver = tf.train.Saver()
        save_path = "model/{0}.ckpt".format(self.name)
        saver.restore(self.sess, save_path)

        # total prediction in each class {Normal, AF, Other, Noise}
        total = {0: 0, 1: 0, 2: 0, 3: 0}
        # total correct prediction in each class {Normal, AF, Other, Noise}
        corrects = {0: 0, 1: 0, 2: 0, 3: 0}

        # run through every single data in test set
        for i in range(self.ecg.ntests):
            # run single forward pass
            X_test, Y_test = self.ecg.get_test(i)

            # no drop out in testing
            loss, logit, summary = self.sess.run([self.loss, self.logits, merged], feed_dict={self.X: X_test, self.Y: Y_test, self.keep_prob: 1})

            # get the prediction
            writter.add_summary(summary, i)
            probs = self.sess.run(tf.nn.softmax(logit))
            pred = self.sess.run(tf.argmax(probs, 1))[0]

            correct = np.argmax(Y_test)

            total[pred] += 1
            if pred == correct:
                corrects[pred] += 1

            if verbose:
                plot(X_test)
                print 'True label is {0}'.format(self.id_to_class_name[correct]), 'The model predicts', self.id_to_class_name[pred]

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
            save_path = "model/{0}.ckpt".format(self.name)
            saver.restore(self.sess, save_path)

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
