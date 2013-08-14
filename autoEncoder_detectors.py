#!/usr/bin/env/python

from RBM import RBM

import scipy.io
import matplotlib.pyplot as plt
import sys

SAMPLE_IMAGE_IDX = 2

def main():
    learningRate = float(sys.argv[1]) if len(sys.argv) >= 2 else 0.0001
    maxIterations = int(sys.argv[2]) if len(sys.argv) >= 3 else 300

    # load data
    data = scipy.io.loadmat('data/usps_resampled.mat')
    train_patterns = data['train_patterns']
    train_labels = data['train_labels']
    test_patterns = data['test_patterns']
    test_labels = data['test_labels']

    # initialize and train RBM
    rbm = RBM(192, train_patterns, learningRate=learningRate, verbose=True)
    rbm.train(convThreshold=0.1, maxIterations=maxIterations)

    print 'Autoencoding. . . '
    hidden_patterns = rbm.translate(train_patterns)
    ae_patterns = rbm.invert(hidden_patterns)
    print 'Finished.'

    W = rbm.W.get_value()
    while True:
        while True:
            try:
                detectorNum = raw_input("Pick a detector image from [0-"+str(W.shape[0]-1)+"] (q to quit): ")
                if detectorNum == 'q':
                    sys.exit(0)
                detectorNum = int(detectorNum)
                if detectorNum not in range(W.shape[0]):
                    raise ValueError
            except ValueError:
                continue
            except EOFError:
                sys.exit(0)
            except KeyboardInterrupt:
                sys.exit(0)
            break

        # show example image
        plt.matshow(W[detectorNum,:].reshape(16, 16))
        plt.show()

if __name__ == '__main__':
    main()
