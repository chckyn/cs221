#!/usr/bin/env/python

from RBM import RBM

import scipy.io
import matplotlib.pyplot as plt

SAMPLE_IMAGE_IDX = 2

def main():
    # load data
    data = scipy.io.loadmat('data/usps_resampled.mat')
    train_patterns = data['train_patterns']
    train_labels = data['train_labels']
    test_patterns = data['test_patterns']
    test_labels = data['test_labels']

    # initialize and train RBM
    rbm = RBM(256, train_patterns, learningRate=0.2)
    rbm.train(convThreshold=0.01)

    print 'Autoencoding. . . '
    hidden_patterns = rbm.translate(train_patterns)
    ae_patterns = rbm.invert(hidden_patterns)
    print 'Finished.'

    # show example image
    plt.matshow(train_patterns[:,SAMPLE_IMAGE_IDX].reshape(16, 16))
    plt.matshow(hidden_patterns[:,SAMPLE_IMAGE_IDX].reshape(16, 16))
    plt.matshow(ae_patterns[:,SAMPLE_IMAGE_IDX].reshape(16, 16))
    plt.show()

if __name__ == '__main__':
    main()
