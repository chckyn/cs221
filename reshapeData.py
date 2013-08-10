#!/usr/bin/env python

import shelve
import scipy.io

from util import Image
import const

IMG_HEIGHT = const.IMG_HEIGHT
IMG_WIDTH = const.IMG_WIDTH
PATCH_HEIGHT = const.PATCH_HEIGHT
PATCH_WIDTH = const.PATCH_WIDTH

def makeImageArray(imageData, imageLabels):
    images = []
    for i in range(imageData.shape[1]):
        patches = [] 
        data = imageData[:,i].reshape(IMG_HEIGHT,IMG_WIDTH)
        for j in range(PATCH_HEIGHT):
            for k in range(PATCH_WIDTH):
                patch = data[j:j+PATCH_HEIGHT,k:k+PATCH_WIDTH]
                patches.append(patch)
    
        label = None
        for j in range(10):
            if imageLabels[j,i] == 1:
                label = j
                break
    
        img = Image(data, tuple(patches), label)
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

