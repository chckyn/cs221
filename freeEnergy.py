#!/usr/bin/env/python

import scipy.io
import math
import numpy
import theano

from numpy import random as rng
from theano import tensor as T

NUM_HID = 128
BINARY_THRESHOLD = 0.5
INIT_WEIGHT_VAR = 1.0
INIT_BIAS_VAR_V = 1.0
INIT_BIAS_VAR_H = 1.0

def initParams(numVis, numHid):
    W = INIT_WEIGHT_VAR*rng.randn(numHid, numVis)
    b_v = INIT_BIAS_VAR_V*rng.randn(numVis)
    b_h = INIT_BIAS_VAR_H*rng.randn(numHid)
    return (W, b_v, b_h)

def makeBinary(v):
    sig = 1.0/(1.0 + numpy.exp(-v))
    out = numpy.zeros_like(v)
    out[sig > BINARY_THRESHOLD] = 1
    return out

def RBM_Free_Energy(x, y):
    # make input data binary
    data = makeBinary(x)

    # determine initial params
    (W_init, b_v_init, b_h_init) = initParams(x.shape[0], NUM_HID)
    W = theano.shared(W_init, name='W')
    b_v = theano.shared(b_v_init, name='b_v')
    b_h = theano.shared(b_h_init.reshape(b_h_init.shape[0], 1), name='b_h')

    # compute free energy
    v = T.matrix('v')
    F = -T.dot(b_v, v) - T.sum(T.log(1.0 + T.exp(T.addbroadcast(b_h, 1) + T.dot(W, v))), axis=0)
    free_energy = theano.function([v], F)
    value = free_energy(data)
    print value.sum()

def main():
    data = scipy.io.loadmat('data/usps_resampled.mat')
    train_patterns = data['train_patterns']
    train_labels = data['train_labels']
    test_patterns = data['test_patterns']
    test_labels = data['test_labels']
    RBM_Free_Energy(train_patterns, train_labels)

if __name__ == '__main__':
    main()
