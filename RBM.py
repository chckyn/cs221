#!/usr/bin/env/python

import sys
import scipy.io
import math
import numpy
import theano

from numpy import random as rng
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

RANDOM_SEED = 1234
NUM_HID = 128
BINARY_THRESHOLD = 0.5
INIT_WEIGHT_SIGMA = 1.0
INIT_VIS_BIAS_SIGMA = 0.0
INIT_HID_BIAS_SIGMA = 0.0
LEARNING_RATE = 0.1
PARAM_CONV_THRESHOLD = 0.01
MAX_TRAIN_ITERATIONS = 1000

def makeBinary(v, threshold=BINARY_THRESHOLD):
    sig = 1.0/(1.0 + numpy.exp(-v))
    out = numpy.zeros_like(v)
    out[sig > threshold] = 1
    return out

def computeVectorRMS(v):
    var_v = v**2
    return math.sqrt(numpy.average(var_v))

def computeMatrixRMS(M):
    var_M = M**2
    return math.sqrt(numpy.average(numpy.average(var_M)))

class RBM(object):
    """Restricted Boltzmann Machine"""

    def __init__(self, numHid, training_data, learningRate=LEARNING_RATE,\
                 randomSeed=RANDOM_SEED, initWeightVar=INIT_WEIGHT_SIGMA,\
                 initVisBiasSigma=INIT_VIS_BIAS_SIGMA, initHidBiasSigma=INIT_HID_BIAS_SIGMA,\
                 verbose=False):
        self.numVis = training_data.shape[0]
        self.numHid = numHid
        self.learningRate = learningRate
        self.verbose = verbose

        # instantiate random number generator
        self.rng = RandomStreams(randomSeed)

        # make training data binary
        self.data = makeBinary(training_data)

        # initialize params
        W_init = initWeightVar*rng.randn(self.numHid, self.numVis)
        self.W = theano.shared(W_init, name='W')
        b_v_init = initVisBiasSigma*rng.randn(self.numVis)
        self.b_v = theano.shared(b_v_init.reshape(b_v_init.shape[0], 1), name='b_v')
        b_h_init = initHidBiasSigma*rng.randn(self.numHid)
        self.b_h = theano.shared(b_h_init.reshape(b_h_init.shape[0], 1), name='b_h')

        # formulate free energy (positive phase)
        v = T.matrix('v')
        F = self.formulateFreeEnergy(v)
        self.free_energy = theano.function([v], F.sum()) # compile method

        # formulate approximate expected free energy
        # using k=1 contrastive divergence (negative phase)
        h_0 = self.sampleHidden(v)
        v_0 = self.sampleVisible(h_0)
        F_exp = self.formulateFreeEnergy(v_0)
        self.exp_free_energy = theano.function([v], F_exp.sum()) # compile method

        # formulate param gradients
        dParams = T.grad(F.sum() - F_exp.sum(),\
            [self.W, self.b_v, self.b_h], consider_constant=[v_0])
        self.dParams_func = theano.function([v], dParams) # compile method

    def translate(self, visData):
        return 1.0/(1.0 + numpy.exp(-self.b_h.get_value() - numpy.dot(self.W.get_value(), visData)))

    def invert(self, hidData):
        return 1.0/(1.0 + numpy.exp(-self.b_v.get_value() - numpy.dot(numpy.transpose(self.W.get_value()), hidData)))

    def sampleHidden(self, v):
        h_mean = 1.0/(1.0 + T.exp(-T.addbroadcast(self.b_h, 1) - T.dot(self.W, v)))
        return self.rng.binomial(size=h_mean.shape, n=1, p=h_mean)

    def sampleVisible(self, h):
        v_mean = 1.0/(1.0 + T.exp(-T.addbroadcast(self.b_v, 1) - T.dot(self.W.T, h)))
        return self.rng.binomial(size=v_mean.shape, n=1, p=v_mean)

    def formulateFreeEnergy(self, v):
        return -T.dot(T.flatten(self.b_v, 1), v) -\
            T.sum(T.log(1.0 + T.exp(T.addbroadcast(self.b_h, 1) + T.dot(self.W, v))), axis=0)

    def train(self, maxIterations=MAX_TRAIN_ITERATIONS, convThreshold=PARAM_CONV_THRESHOLD):
        old_free_energy = self.free_energy(self.data)
        old_exp_free_energy = self.exp_free_energy(self.data)

        for i in range(maxIterations):
            # compute gradients
            (dW, db_v, db_h) = self.dParams_func(self.data)

            # compute RMS values
            W_rms = computeMatrixRMS(self.W.get_value())
            dW_rms = computeMatrixRMS(dW)
            b_v_rms = computeVectorRMS(self.b_v.get_value())
            db_v_rms = computeVectorRMS(db_v)
            b_h_rms = computeVectorRMS(self.b_h.get_value())
            db_h_rms = computeVectorRMS(db_h)

            # update params
            self.W.set_value(self.W.get_value() + self.learningRate*dW)
            self.b_v.set_value(self.b_v.get_value() + self.learningRate*db_v)
            self.b_h.set_value(self.b_h.get_value() + self.learningRate*db_h)

            # compute free energy
            new_free_energy = self.free_energy(self.data)
            new_exp_free_energy = self.exp_free_energy(self.data)
            delta_free_energy = new_free_energy - old_free_energy
            delta_exp_free_energy = new_exp_free_energy - old_exp_free_energy

            if self.verbose:
                print '===============', i, '================'

                # print params and their gradients
                if W_rms > 0 and b_v_rms > 0 and b_h_rms > 0:
                    print '   <W> =', '{:.2e}'.format(float(W_rms)),\
                          '  <dW> =', '{:.2e}'.format(float(dW_rms)),\
                          'delta =', '{:.3f}'.format(float(dW_rms/W_rms))
                    print ' <b_v> =', '{:.2e}'.format(float(b_v_rms)),\
                          '<db_v> =', '{:.2e}'.format(float(db_v_rms)),\
                          'delta =', '{:.3f}'.format(float(db_v_rms/b_v_rms))
                    print ' <b_h> =', '{:.2e}'.format(float(b_h_rms)),\
                          '<db_h> =', '{:.2e}'.format(float(db_h_rms)),\
                          'delta =', '{:.3f}'.format(float(db_h_rms/b_h_rms))
 
                # print free energy
                print '                    F =', '{:.2e}'.format(float(old_free_energy))
                print '              Delta F =', '{:.2e}'.format(float(delta_free_energy))
                print '      Est. Expected F =', '{:.2e}'.format(float(old_exp_free_energy))
                print 'Delta Est. Expected F =', '{:.2e}'.format(float(delta_exp_free_energy))

            # check for convergence
            if W_rms > 0 and dW_rms/W_rms < convThreshold and\
               b_v_rms > 0 and db_v_rms/b_v_rms < convThreshold and\
               b_h_rms > 0 and db_h_rms/b_h_rms < convThreshold:
                break

            old_free_energy = new_free_energy
            old_exp_free_energy = new_exp_free_energy

def main():
    # load data
    data = scipy.io.loadmat('data/usps_resampled.mat')
    train_patterns = data['train_patterns']
    train_labels = data['train_labels']
    test_patterns = data['test_patterns']
    test_labels = data['test_labels']

    # initialize and train RBM
    rbm = RBM(NUM_HID, train_patterns, learningRate=0.2, verbose=True)
    rbm.train(convThreshold=0.05)

if __name__ == '__main__':
    main()
