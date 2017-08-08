import argparse, math, random
import torch as th
import torchnet as tnt

import loader, models, optim
import numpy as np
import pdb, glob, sys, os

bsz = 1024
L = 1
n = 3
maxb = 500

opt = dict(b=bsz, frac=1.0, n=3, m='mnist', augment=True, nw=1)
d, augment = getattr(loader, opt['m'])(opt)

class DS(object):
    def __init__(self, d):
        self.d = d
        self.n = d['x'].size(0)

    def __getitem__(self, idx):
        i = idx % self.n
        return (self.d['x'][i], self.d['y'][i])

    def __len__(self):
        return 2**20

ds = [th.utils.data.DataLoader(DS(d['train']), batch_size=opt['b']) for _ in xrange(opt['n'])]

# for e in xrange(100):
#     for bi, (x,y) in enumerate(ds[0]):
#         print e, bi
#         if bi > 500:
#             break

# option 1
# loaders = loader.get_loaders(d, augment, opt)
# ds = loaders[0]['train']
# iters = [ds[i].__iter__() for i in xrange(opt['n'])]
# for e in xrange(100):
#     for bi in xrange(maxb):
#         for l in xrange(L):
#             for ni in xrange(opt['n']):
#                 try:
#                     x,y = next(iters[i])
#                 except StopIteration:
#                     iters[i] = ds[i].__iter__()
#                     x,y = next(iters[i])
#         print e, bi

m = models.lenet({'d': 0.25})
n = models.num_parameters(m)
t = th.FloatTensor(n)
x, dx = t.clone(), t.clone()
optim.flatten_params(m, x, dx)

for bi, (xi,ti) in enumerate(ds[0]):
    tih = m(xi)
    f = nn.CrossEntropyLoss()(tih, ti)
    f.backward()
    
    print dx[:25].view(5,5)
    print list(m.parameters())[0].grad[0]

