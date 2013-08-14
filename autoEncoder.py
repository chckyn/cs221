#!/usr/bin/env/python

from RBM import RBM

import numpy
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

    y = raw_input("Load RBM params (n)? ")
    if y in ['y', 'yes']:
        name = raw_input("Name of params: ")
        W_file = open('data/{0}.W.npy'.format(name))
        rbm.W.set_value(numpy.load(W_file))
        W_file.close()
        b_v_file = open('data/{0}.b_v.npy'.format(name))
        rbm.b_v.set_value(numpy.load(b_v_file))
        b_v_file.close()
        b_h_file = open('data/{0}.b_h.npy'.format(name))
        rbm.b_h.set_value(numpy.load(b_h_file))
        b_h_file.close()
    else:
        rbm.train(convThreshold=0.1, maxIterations=maxIterations)

    print 'Autoencoding. . . '
    hidden_patterns = rbm.translate(train_patterns)
    ae_patterns = rbm.invert(hidden_patterns)
    print 'Finished.'

    while True:
        while True:
            try:
                sampleImage = raw_input("Pick a sample image from [0-"+str(train_patterns.shape[1]-1)+"] (q to quit): ")
                if sampleImage == 'q':
                    y = raw_input("Save RBM params (y)? ")
                    if y not in ['n', 'no']:
                        name = raw_input("Name these params: ")
                        W_file = open('data/{0}.W.npy'.format(name), 'w')
                        numpy.save(W_file, rbm.W.get_value())
                        W_file.close()
                        b_v_file = open('data/{0}.b_v.npy'.format(name), 'w')
                        numpy.save(b_v_file, rbm.b_v.get_value())
                        b_v_file.close()
                        b_h_file = open('data/{0}.b_h.npy'.format(name), 'w')
                        numpy.save(b_h_file, rbm.b_h.get_value())
                        b_h_file.close()
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
