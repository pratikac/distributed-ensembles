import argparse, math, random
import torch as th

import loader
import numpy as np
import pdb, glob, sys, os

bsz = 1024
L = 1
n = 3

opt = dict(b=bsz, frac=0.5, n=n, m='cifar10', augment=True, nw=4)
d, augment = getattr(loader, opt['m'])(opt)
loaders = loader.get_loaders(d, augment, opt)
ds = loaders[0]['train']

# option 1
dsiter = ds.__iter__()
for e in xrange(10):
    maxb = len(ds)
    print 'maxb: ', maxb
    for bi in xrange(maxb):
        for l in xrange(L):
            try:
                x,y = next(dsiter)
            except StopIteration:
                dsiter = ds.__iter__()
                x,y = next(dsiter)
        print e, bi