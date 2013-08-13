#!/usr/bin/env/python

import scipy.io
import math
import numpy
import theano

from numpy import random as rng
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

NUM_HID = 128
BINARY_THRESHOLD = 0.5
INIT_WEIGHT_VAR = 1.0
INIT_BIAS_VAR_V = 1.0
INIT_BIAS_VAR_H = 1.0
RANDOM_SEED = 1234

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
    b_v = theano.shared(b_v_init.reshape(b_v_init.shape[0], 1), name='b_v')
    b_h = theano.shared(b_h_init.reshape(b_h_init.shape[0], 1), name='b_h')

    # compute free energy
    v = T.matrix('v')
    F = -T.dot(T.flatten(b_v, 1), v)\
        - T.sum(T.log(1.0 + T.exp(T.addbroadcast(b_h, 1) + T.dot(W, v))), axis=0)
    free_energy = theano.function([v], F.sum())
    value = free_energy(data)
    print '             Total Free Energy =', value

    # approximate expected free energy
    # using k=1 constrastive divergence
    rng = RandomStreams(RANDOM_SEED)
    h_0_mean = 1.0/(1.0 + T.exp(-T.addbroadcast(b_h, 1) - T.dot(W, v)))
    h_0 = rng.binomial(size=h_0_mean.shape, n=1, p=h_0_mean)
    v_0_mean = 1.0/(1.0 + T.exp(-T.addbroadcast(b_v, 1) - T.dot(W.T, h_0)))
    v_0 = rng.binomial(size=v_0_mean.shape, n=1, p=v_0_mean)
    F_exp = -T.dot(T.flatten(b_v, 1), v_0)\
            - T.sum(T.log(1.0 + T.exp(T.addbroadcast(b_h, 1) + T.dot(W, v_0))), axis=0)
    exp_free_energy = theano.function([v], F_exp.sum())
    value = exp_free_energy(data)
    print 'Estimated Expected Free Energy =', value

    # compute param deltas
    dParams = T.grad(F.sum() - F_exp.sum(), [W, b_v, b_h], consider_constant=[v_0])
    dParams_func = theano.function([v], dParams)
    value = dParams_func(data)
    #print value

def main():
    data = scipy.io.loadmat('data/usps_resampled.mat')
    train_patterns = data['train_patterns']
    train_labels = data['train_labels']
    test_patterns = data['test_patterns']
    test_labels = data['test_labels']
    RBM_Free_Energy(train_patterns, train_labels)

if __name__ == '__main__':
    main()
