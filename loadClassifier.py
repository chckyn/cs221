#!/usr/bin/env python

import pickle
import sys
from matplotlib import pyplot as plt
import scipy

f = open(sys.argv[1])
rbm = pickle.load(f)
data = scipy.io.loadmat('data/usps_resampled.mat')
train_patterns = data['train_patterns']

while True:
    while True:
        try:
            sampleImage = raw_input("Pick a sample image from [0-"+str(train_patterns.shape[1]-1)+"] (q to quit): ")
            if sampleImage == 'q':
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
    #
    plt.matshow(train_patterns[:,sampleImage].reshape(16, 16))
    plt.matshow(rbm.invert(rbm.translate(train_patterns[:,sampleImage].reshape(256,1))).reshape(16, 16))
    
    plt.show()
