#!/usr/bin/env python

import shelve

db = shelve.open('data/uspsDigit.db')
trainImages=db['trainImages']
testImages=db['testImages']

print trainImages[1239].label
trainImages[1239].view()
print testImages[193].label
testImages[193].view()

