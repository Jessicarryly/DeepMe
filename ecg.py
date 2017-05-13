from os.path import join
import numpy as np
from csv import reader
from scipy.io import loadmat
from sklearn.cross_validation import train_test_split, check_random_state

class ECG:
    """
    ECG is the container for loading Electrocardiogram (ECG) data from the
    2017 physionet challenge
    """

    def __init__(self, training_path='training2017', validation_path='validation', csvfile='REFERENCE.csv', random_state=69):
        self.training_path = training_path
        self.validation_path = validation_path
        self.class_name_to_id = {'N': 0, 'A': 1, 'O': 2, '~': 3}
        self.nclasses = len(self.class_name_to_id.keys())
        self.csvfile = csvfile
        self.random_state = random_state


        self.X_train = None
        self.Y_train = None

        self.X_test = None
        self.Y_test = None

        # 3002: min validation
        # 2714: min training
        self.minlen = 2714
        # 18170: max validation
        # 18286: max train
        self.maxlen = 18286
        self.__init_dataset()



    def __init_dataset(self):
        self.__load_training_data()
        self.__load_validation_data()


    def __load_training_data(self):
        self.X_train, self.Y_train = self.__load_data(self.training_path, self.csvfile)


    def __load_validation_data(self):
        self.X_test, self.Y_test = self.__load_data(self.validation_path, self.csvfile)
        self.ntests = self.X_test.shape[0]

    def __load_data(self, path, csvfile):
        X = []
        Y = []
        with open(join(path, csvfile), 'rb') as file:
            ecgreader = reader(file, delimiter=',')
            for row in ecgreader:
                val = loadmat(join(path, row[0]))['val']
                label = row[1]
                X.append(val[:, 0:self.minlen].reshape(self.minlen))
                Y.append(self.class_name_to_id[label])
        X = np.asarray(X)
        Y = np.asarray(Y)
        Y = np.eye(self.nclasses)[Y]
        return X, Y

    def get_train_batch(self, batch_size):
        """
        Helper function to get mini batch from the training set
        """
        n, d = self.X_train.shape
        idx = np.random.choice(range(n), batch_size, replace=False)
        return self.X_train[idx,:], self.Y_train[idx]

    def get_test_batch(self, batch_size):
        """
        Helper function to get mini batch from the test set
        """
        n, d = self.X_test.shape
        idx = np.random.choice(range(n), batch_size, replace=False)
        return self.X_test[idx, :], self.Y_test[idx]

def testECG():
    ecg = ECG()
    ecg.get_mini_batch(25)
