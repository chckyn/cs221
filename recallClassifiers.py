#!/usr/bin/env python

import shelve
import sys
from matplotlib import pyplot as plt

db = shelve.open('data/classifiers.db')
c = db[sys.argv[1]]

while True:
    while True:
        try:
            sampleImage = raw_input("Pick a sample image from [0-"+str(c.train_patterns.shape[1]-1)+"] (q to quit): ")
            if sampleImage == 'q':
                sys.exit(0)
            sampleImage = int(sampleImage)
            if sampleImage not in range(c.train_patterns.shape[1]):
                raise ValueError
        except ValueError:
            continue
        except EOFError:
            sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)
        break

    # show example image
    plt.matshow(c.train_patterns[:,sampleImage].reshape(16, 16))
    plt.matshow(c.binary_train_patterns[:,sampleImage].reshape(16, 16))
    plt.matshow(c.hidden_patterns[:,sampleImage].reshape(16, 16))
    plt.matshow(c.ae_patterns[:,sampleImage].reshape(16, 16))
    plt.show()
