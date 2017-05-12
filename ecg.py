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

    def __init__(self, basepath='validation', csvfile='REFERENCE.csv', random_state=69):
        self.basepath = basepath
        self.class_name_to_id = {'N': 0, 'A': 1, 'O': 2, '~': 3}
        self.nclasses = len(self.class_name_to_id.keys())
        self.csvfile = csvfile
        self.random_state = random_state

        self.X = []
        self.Y = []

        self.X_train = []
        self.Y_train = []

        self.X_test = []
        self.Y_test = []

        # 3002: min validation
        # 2714: min training
        self.minlen = 2714
        # 18170: max validation
        # 18286: max train
        self.maxlen = 18286
        self.__init_dataset()



    def __init_dataset(self):
        self.__load_data()
        self.__split_data()


    def __load_data(self):
        with open(join(self.basepath, self.csvfile), 'rb') as file:
            ecgreader = reader(file, delimiter=',')
            for row in ecgreader:
                self.__parse_data(row)
    
        self.X = np.asarray(self.X)
        self.Y = np.asarray(self.Y)
        self.Y = np.eye(self.nclasses)[self.Y]


    def __parse_data(self, row):
        val = loadmat(join(self.basepath, row[0]))['val']
        label = row[1]
        self.Y.append(self.class_name_to_id[label])
        self.X.append(val[:,0:self.minlen].reshape(self.minlen))

    def __split_data(self):
        """
        Randomly split the data into trainning set and test set
        """

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=self.random_state)
        print 'Train:', self.X_train.shape, self.Y_train.shape
        print 'Test: ', self.X_test.shape, self.Y_test.shape
        print 'Whole:', self.X.shape, self.Y.shape
        self.ntrains = self.X_train.shape[0]
        self.ntests = self.X_test.shape[0]

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
