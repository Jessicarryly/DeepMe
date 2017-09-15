from solver import Solver
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
            solver = Solver(ecg=ecg)
            solver.train()
        elif mode == 'test':
            solver = Solver()
            solver.test(sample_every=30)
        elif mode == 'develop':
            ecg = ECG(verbose=True)
            solver = Solver(ecg=ecg, develop=True)
            solver.train
        else:
            solver = Solver()
            solver.predict(mode)
    else:
        print("usage: python deepme.py <option>")
