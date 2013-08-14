#!/usr/bin/env/python

from RBM import RBM
from RBM import makeBinary

import scipy.io
import matplotlib.pyplot as plt
import sys
from copy import deepcopy

from classifier import Classifier
import shelve

SAMPLE_IMAGE_IDX = 2

def main():
    learningRate = float(sys.argv[1]) if len(sys.argv) >= 2 else 1e-16
    maxIterations = int(sys.argv[2]) if len(sys.argv) >= 3 else 1000

    # load data
    data = scipy.io.loadmat('data/usps_resampled.mat')
    train_patterns = data['train_patterns']
    train_labels = data['train_labels']
    test_patterns = data['test_patterns']
    test_labels = data['test_labels']

    binary_train_patterns = deepcopy(train_patterns)
    for i in range(train_patterns.shape[1]):
        binary_train_patterns[:,i] = makeBinary(train_patterns[:,i])

    # initialize and train RBM
    rbm = RBM(256, train_patterns, learningRate=learningRate, verbose=True)
    iterationsCompleted = rbm.train(convThreshold=0.01, maxIterations=maxIterations)

    print 'Autoencoding. . . '
    hidden_patterns = rbm.translate(train_patterns)
    hidden_patterns = rbm.translate(train_patterns)
    ae_patterns = rbm.invert(hidden_patterns)
    print 'Finished.'

    while True:
        while True:
            try:
                sampleImage = raw_input("Pick a sample image from [0-"+str(train_patterns.shape[1]-1)+"] (q to quit): ")
                if sampleImage == 'q':
                    y = raw_input("Save this classifier (y)? ")
                    if y in ['y','']:
                        db = shelve.open('data/classifiers.db')
                        db[str((learningRate, 256, iterationsCompleted))] = \
                                Classifier(train_patterns, binary_train_patterns, hidden_patterns, ae_patterns)
                        db.close()
                    sys.exit(0)
                sampleImage = int(sampleImage)
                if sampleImage not in range(train_patterns.shape[1]):
                    raise ValueError
            except ValueError:
                continue
            except EOFError:
                sys.exit(0)
            except KeyboardInterrupt:
                sys.exit(0)
            break

        # show example image
        plt.matshow(train_patterns[:,sampleImage].reshape(16, 16))
        plt.matshow(binary_train_patterns[:,sampleImage].reshape(16, 16))
        plt.matshow(hidden_patterns[:,sampleImage].reshape(16, 16))
        plt.matshow(ae_patterns[:,sampleImage].reshape(16, 16))
        plt.show()


if __name__ == '__main__':
    main()