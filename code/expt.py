import argparse, math, random
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import models, loader, optim
import numpy as np
from pprint import pprint
import pdb, glob, sys, os

bsz = 1024
L = 1
n = 3

ds = th.utils.data.DataLoader(
    datasets.MNIST('/local2/pratikac/mnist', train=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=bsz, shuffle=True,
    num_workers=0, pin_memory=True)

# option 0
# for e in xrange(10):
#     for bi, (x,y) in enumerate(ds):
#         print e, bi


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