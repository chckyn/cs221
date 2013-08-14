#!/usr/bin/env/python

from RBM import RBM

import numpy
import scipy.io
import matplotlib.pyplot as plt
import sys
from copy import deepcopy

from classifier import Classifier
import pickle

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
    iterationsCompleted = rbm.train(convThreshold=0.03, maxIterations=maxIterations)

    print 'Autoencoding. . . '
    hidden_patterns = rbm.translate(train_patterns)
    ae_patterns = rbm.invert(hidden_patterns)
    print 'Finished.'

    while True:
        while True:
            try:
                sampleImage = raw_input("Pick a sample image from [0-"+str(train_patterns.shape[1]-1)+"] (q to quit): ")
                if sampleImage == 'q':
                    y = raw_input("Save this classifier (y)? ")
                    fn = 'data/classifier_'+str((learningRate, 192, iterationsCompleted))
                    if y in ['y','']:
                        f = open(fn,'w')
                        pickle.dump(Classifier(train_patterns, binary_train_patterns, hidden_patterns, ae_patterns), f)
                        print "Classifer saved as "+fn
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
        plt.matshow(ae_patterns[:,sampleImage].reshape(16, 16))
        plt.show()

if __name__ == '__main__':
    main()
