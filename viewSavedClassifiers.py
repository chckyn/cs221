#!/usr/bin/env python

import shelve

db = shelve.open('data/classifiers.db')
print db.keys()
