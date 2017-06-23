from __future__ import print_function
import argparse, math, random
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import torch.nn.functional as F
from torch.nn.parallel import scatter, parallel_apply, gather

from exptutils import *
import models, loader, optim
import numpy as np
import logging
from pprint import pprint
import pdb, glob, sys
from collections import OrderedDict
from copy import deepcopy

opt = add_args([
['-g', 3, 'gpu idx'],
['-b', 20, 'batch_size'],
['-e', 0, 'start epoch'],
['-B', 6, 'Max epochs'],
['-T', 35, 'bptt'],
['--lr', 20.0, 'learning rate'],
['--lrs', '', 'learning rate schedule'],
['--clip', 0.25, 'gradient clipping'],
['-s', 1111, 'seed']
])

th.manual_seed(opt['s'])
th.cuda.manual_seed(opt['s'])

corpus, ptb, batcher = loader.ptb(opt)
opt['vocab'] = len(corpus.dictionary)

model = models.ptbs(opt).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = th.optim.SGD(model.parameters(), lr=opt['lr'])

def train(e):
    model.train()

    dt = AverageMeter()
    f = 0

    bsz = opt['b']
    maxb = (ptb['train'].size(0) -1) // opt['T']
    t0 = timer()

    h = model.init_hidden(opt['b'])
    for bi, idx in enumerate(range(0, ptb['train'].size(0) - 1, opt['T'])):
        _dt = timer()
        x, y = batcher(ptb['train'], idx)
        x, y = Variable(x.cuda()), Variable(y.squeeze().cuda())

        _h = models.repackage_hidden(h)
        model.zero_grad()
        yh, hh = model(x, _h)
        _f = criterion(yh.view(-1, opt['vocab']), y)
        _f.backward()

        nn.utils.clip_grad_norm(model.parameters(), opt['clip'])
        for p in model.parameters():
            p.data.add_(-opt['lr'], p.grad.data)

        f += _f.data[0]
        dt.update(timer()-_dt, 1)

        bif = 200
        if bi % bif == 0 and bi != 0:
            f = f/float(bif)
            print((color('blue', '[%2.2fs][%2d][%4d/%4d] %2.4f %2.2f'))%(dt.avg,
                e,bi,maxb, f, math.exp(f)))
            f = 0

    print()

def val(e, src):
    model.eval()

    bsz = opt['b']
    h = model.init_hidden(bsz)
    f = 0.0

    for i in range(0, ptb[src].size(0)-1, opt['T']):
        x,y = batcher(ptb[src], i)

        x,y = Variable(x.cuda(), volatile=True), \
                Variable(y.squeeze().cuda(), volatile=True)

        h = models.repackage_hidden(h)
        yh,hh = model(x, h)
        _f = criterion(yh.view(-1, opt['vocab']), y).data[0]
        f = f + _f*len(x)
        print(i, _f, len(x))

    f = f/len(ptb[src])
    print((color('red', '**[%2d] %2.4f %2.4f\n'))%(e, f, math.exp(f)))

for e in xrange(opt['e'], opt['B']):
    train(e)
    val(e, 'valid')
val(opt['B']-1, 'test')
