from model import CNN
from ecg import ECG
import sys

if __name__ == '__main__':
    """
    the cli program to run the model
    usage: python deepme.py <option>
    options: train -> train the new model
             develop -> develop the better model
             test -> run the test on the whole data set
             path/to/mat -> Run the prediction to .mat file

    """

    if len(sys.argv) == 2:
        mode = sys.argv[1]
        if mode == 'train':
            ecg = ECG(verbose=True)
            model = CNN(ecg=ecg)
            model.train()
        elif mode == 'test':
            model = CNN()
            model.test(sample_every=30)
        elif mode == 'develop':
            ecg = ECG(verbose=True)
            model = CNN(ecg=ecg, develop=True)
            model.train
        else:
            model = CNN()
            model.predict(mode)
    else:
        print "usage: python deepme.py <option>"
