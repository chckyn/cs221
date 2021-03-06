#!/usr/bin/env python

import shelve
import scipy.io
from util import Image

def makeImageArray(imageData, imageLabels):
    images = []
    for i in range(imageData.shape[1]):
        pixels = imageData[:,i]
    
        label = None
        for j in range(10):
            if imageLabels[j,i] == 1:
                label = j
                break
        assert not label == None
    
        img = Image(pixels, label)
        images.append(img)

    return images

data = scipy.io.loadmat('data/usps_resampled.mat')

train_patterns = data['train_patterns']
train_labels   = data['train_labels']
test_patterns  = data['test_patterns']
test_labels    = data['test_labels']

trainImages = makeImageArray(train_patterns, train_labels)
testImages  = makeImageArray(test_patterns, test_labels)

db = shelve.open('data/uspsDigit.db')
db['trainImages'] = trainImages
db['testImages']  = testImages
db.close()

