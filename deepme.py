from model import VGG
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
            model = VGG(ecg=ecg)
            model.train()
        elif mode == 'test':
            model = VGG()
            model.test(sample_every=30)
        elif mode == 'develop':
            ecg = ECG(verbose=True)
            model = VGG(ecg=ecg, develop=True)
            model.train
        else:
            model = VGG()
            model.predict(mode)
    else:
        print "usage: python deepme.py <option>"
