from os.path import join
import numpy as np
from csv import reader
from scipy.io import loadmat
from sklearn.cross_validation import train_test_split, check_random_state
import matplotlib.pyplot as plt
import gnuplotlib as gp

class ECG:
    """
    ECG is the wrapper for loading Electrocardiogram (ECG) data from the
    2017 physionet challenge and feed it into training model
    """

    def __init__(self, training_path='training2017', validation_path='validation', csvfile='REFERENCE.csv', random_state=69, use_all_feature=False, percent_data_use=10):
        """
        setting up the data model
        training_path: path to training directory
        validation_path: path to validation directory
        csvfile: file that contain the label for each data
        use_all_feature: since the len of the data are vary and may waste time to feed all of them into the model,
            I decide to remove all unnecessary value and cut the data according to the minimum len in the whole dataset
            If use all, those data where the len < max len will be padding with 0
        percent_data_use: since the data set is quite large and take a lot of time just to validate if a small update worth,
            I decide to only take x% of the data for that just purpose
        """
        self.training_path = training_path
        self.validation_path = validation_path
        self.class_name_to_id = {'N': 0, 'A': 1, 'O': 2, '~': 3}
        self.nclasses = len(self.class_name_to_id.keys())
        self.csvfile = csvfile
        self.use_all_feature = use_all_feature
        self.percent_data_use = percent_data_use
        self.random_state = random_state

        # training and testing dataset
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        # 3002: min validation len
        # 2714: min training len
        # 18170: max validation len
        # 18286: max training len
        self.minlen = 2714 # cut dataset to this size if not use all
        self.maxlen = 18286 # max len of the data, padding with 0 if necessary

        # number of feature in each data
        if self.use_all_feature:
            self.nfeatures = self.maxlen
        else:
            self.nfeatures = self.minlen

        self.__init_dataset()



    def __init_dataset(self):
        # load training dataset
        self.X_train, self.Y_train = self.__setup_data(self.training_path)
        self.ntrains = self.X_train.shape[0]
        print 'Train ', self.X_train.shape

        # load testing dataset
        self.X_test, self.Y_test = self.__setup_data(self.validation_path)
        self.ntests = self.X_test.shape[0]
        print 'Test ', self.X_test.shape

        # plot 4 example graph in the dataset
        self.__graph_sample_data()

    def __setup_data(self, path):
        X, Y = self.__load_data(path, self.csvfile)
        X, Y = self.__preprocess_data(X, Y)
        return X, Y

    def __load_data(self, path, csvfile):
        X = []
        Y = []
        with open(join(path, csvfile), 'rb') as file:
            ecgreader = reader(file, delimiter=',')
            for row in ecgreader:
                val = loadmat(join(path, row[0]))['val']
                label = row[1]
                if self.use_all_feature:
                    x = np.zeros(self.maxlen)
                    n = val.shape[1]
                    val = val.reshape(n)
                    x[0:n] = val
                    X.append(x)
                else:
                    X.append(val[:, 0:self.minlen].reshape(self.minlen))
                Y.append(self.class_name_to_id[label])

        # convert of python list to ndarray
        X = np.asarray(X)
        Y = np.asarray(Y)

        # only take a subset of the data
        if self.percent_data_use < 100:
            N, d = X.shape
            n = N / self.percent_data_use
            idx = np.random.choice(range(N), n, replace=False)
            X = X[idx, :]
            Y = Y[idx]

        Y = np.eye(self.nclasses)[Y]

        return X, Y

    # TODO: extract to utils file
    def __preprocess_data(self, X, Y):
        """
        preprocess the data with fft, filter, pankin tomson, wavelet, whatever work
        """
        return X, Y


    def __graph_sample_data(self):
        """
        graph for ecg signal present in the validation
        """
        self.__plot_mat('A00001', 'Normal', self.validation_path)
        self.__plot_mat('A00004', 'AF', self.validation_path)
        self.__plot_mat('A00011', 'Other', self.validation_path)
        self.__plot_mat('A00585', 'Noise', self.validation_path)

    # TODO: extract to utils file
    def __plot_mat(self, file, title, path):
        val = loadmat(join(path, file))['val'][:, 0:self.minlen]
        gp.plot(val, title=title, xlabel='Time (s)', ylabel='Amplitude (mV)', _with='lines', terminal='dumb 120, 40', unset='grid')
        # gp.xlabel('Time (s)')
        # gp.ylabel('Amplitude (mV)')
        # gp.title(title)
        # plt.show()

    def get_train_batch(self, batch_size):
        """
        Helper function to get mini batch from the training set for each iteration during training
        """
        n, d = self.X_train.shape
        idx = np.random.choice(range(n), batch_size, replace=False)
        return self.X_train[idx,:], self.Y_train[idx]

    def get_test_batch(self, batch_size):
        """
        Helper function to get mini batch from the test set for testing
        """
        n, d = self.X_test.shape
        idx = np.random.choice(range(n), batch_size, replace=False)
        return self.X_test[idx, :], self.Y_test[idx]
