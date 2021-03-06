import numpy as np
from utils import *

class ECG:
    """
    ECG is the wrapper for loading Electrocardiogram (ECG) data from the
    2017 physionet challenge and feed it into training model
    """

    def __init__(self, training_path='training2017', validation_path='validation', csvfile='REFERENCE.csv', random_state=69, use_all_feature=False, percent_data_use=100, verbose=False):
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
        self.verbose = verbose

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
            self.nfeatures = 16384 # 2^14
        else:
            self.nfeatures = 2048 # 2^11

        self.__init_dataset()


    def __init_dataset(self):
        # load training dataset
        self.X_train, self.Y_train, _ = self.__setup_data(self.training_path)
        self.ntrains = self.X_train.shape[0]

        # load testing dataset
        self.X_test, self.Y_test, count = self.__setup_data(self.validation_path)

        self.ntests = self.X_test.shape[0]
        self.N, self.A, self.O, self.P = count

        # plot 4 example graph in the dataset
        if self.verbose:
            print('Train ', self.X_train.shape)
            print('Test ', self.X_test.shape)
            print('Label each', count)
            self.__graph_sample_data()

    def __setup_data(self, path):
        X, Y, count = load_data(path=path,
                                csvfile=self.csvfile,
                                percent=self.percent_data_use,
                                all_feature=self.use_all_feature,
                                ids=self.class_name_to_id)
        X = preprocess_data(X)
        return X, Y, count

    def __graph_sample_data(self):
        """
        graph for ecg signal present in the validation
        """
        plot_mat('A00001', 'Normal', self.validation_path)
        plot_mat('A00004', 'AF', self.validation_path)
        plot_mat('A00011', 'Other', self.validation_path)
        plot_mat('A00585', 'Noise', self.validation_path)


    def get_train_batch(self, batch_size):
        """
        Helper function to get mini batch from the training set for each iteration during training
        """
        n, d = self.X_train.shape
        idx = np.random.choice(range(n), batch_size, replace=False)
        return self.X_train[idx], self.Y_train[idx]

    def get_test(self, position):
        # [[postion]], so that out array dont reshape to vector
        return self.X_test[[position]], self.Y_test[[position]]
