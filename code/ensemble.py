from __future__ import print_function
import argparse, math, random
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from exptutils import *
import models, loader, optim
import numpy as np
import logging
from pprint import pprint
import pdb, glob, sys
from collections import OrderedDict
from copy import deepcopy

opt = add_args([
['-o', '/local2/pratikac/results', 'output'],
['-m', 'lenet', 'lenet | mnistfc | allcnn | wideresnet'],
['--optim', 'SGD', 'ESGD | HJB | SGLD | SGD | HEAT'],
['--dataset', 'mnist', 'mnist | rotmnist | cifar10 | cifar100'],
['--frac', 1.0, 'fraction of dataset'],
['-b', 128, 'batch_size'],
['--augment', False, 'data augmentation'],
['-e', 0, 'start epoch'],
['-B', 100, 'Max epochs'],
['--lr', 0.1, 'learning rate'],
['--lrs', '', 'learning rate schedule'],
['--l2', 0.0, 'ell-2'],
['-d', 0.0, 'dropout'],
['-n', 1, '#replicas'],
['-L', 0, 'sgld iterations'],
['--eps', 1e-4, 'sgld noise'],
['--g0', 1e-4, 'gamma'],
['--g1', 0.0, 'scoping'],
['-s', 42, 'seed'],
['-l', False, 'log'],
['-f', 10, 'print freq'],
['-t', 4, 'num threads'],
['-v', False, 'verbose'],
['--validate', '', 'validate a checkpoint'],
['--validate_ensemble', '', 'validate an ensemble'],
['--save', False, 'save network']
])
if opt['L'] > 0:
    opt['f'] = 1
if opt['l']:
    opt['f'] = 1

gpus = [0,1,2]
th.set_num_threads(2)
random.seed(opt['s'])
np.random.seed(opt['s'])
th.manual_seed(opt['s'])
th.cuda.manual_seed_all(opt['s'])
cudnn.benchmark = True

build_filename(opt, blacklist=['lrs','retrain','step', \
                            'f','v','dataset', 'augment', 'd', 't',
                            'depth', 'widen','save','e','validate','l2','eps',
                            'validate_ensemble'])
logger = create_logger(opt)
pprint(opt)

model = models.ReplicateModel(getattr(models, opt['m'])(opt), \
            nn.CrossEntropyLoss(), opt['n'], gpus)
train_loader, val_loader, test_loader = getattr(loader, opt['dataset'])(opt)

optimizer = optim.ElasticSGD(model.ensemble[0].parameters(),
        config = dict(lr=opt['lr'], momentum=0.9, nesterov=True, weight_decay=opt['l2'],
        L=opt['L'], eps=opt['eps'], g0=opt['g0'], g1=opt['g1'], verbose=opt['v'])
        )

def train(e):
    model.train()

    f, top1, dt = AverageMeter(), AverageMeter(), AverageMeter()

    bsz = opt['b']
    maxb = len(train_loader)
    t0 = timer()

    for bi in xrange(maxb):
        _dt = timer()

        def helper():
            def feval():
                xs, ys = [], []
                for i in xrange(opt['n']):
                    x, y = next(train_loader)
                    x, y =  Variable(x.cuda(model.gidxs[i])), \
                            Variable(y.squeeze().cuda(model.gidxs[i]))
                    xs.append(x)
                    ys.append(y)

                fs, errs = model(xs, ys)
                model.backward()
                fs = [fs[i].data[0] for i in xrange(opt['n'])]
                return fs, errs
            return feval

        fs, errs = optimizer.step(helper(), model)

        f.update(np.mean(fs), bsz)
        top1.update(np.mean(errs), bsz)
        dt.update(timer()-_dt, 1)

        if bi % 25 == 0:
            print((color('blue', '[%2.2fs][%2d][%4d/%4d] %2.4f %2.2f%%'))%(dt.avg, e,bi,maxb,
                f.avg, top1.avg))

    print((color('blue', '++[%2d] %2.4f %2.2f%% [%2.2fs]'))% (e,
        f.avg, top1.avg, timer()-t0))
    print()

def val(e):
    def set_dropout(m, cache = None, p=0):
        if cache is None:
            cache = []
            for l in m.modules():
                if 'Dropout' in str(type(l)):
                    cache.append(l.p)
                    l.p = p
            return cache
        else:
            for l in m.modules():
                if 'Dropout' in str(type(l)):
                    assert len(cache) > 0, 'cache is empty'
                    l.p = cache.pop(0)

    def dry_feed(m):
        m.train()
        cache = set_dropout(m)
        maxb = len(train_loader)
        for bi in xrange(maxb):
            x,y = next(train_loader)
            x,y =   Variable(x.cuda(0), volatile=True), \
                    Variable(y.squeeze().cuda(0), volatile=True)
            yh = m(x)
        set_dropout(m,cache)

    dry_feed(model.reference)
    model.eval()

    maxb = len(val_loader)
    f, top1 = AverageMeter(), AverageMeter()

    for bi in xrange(maxb):
        x,y = next(val_loader)
        bsz = x.size(0)

        # xs, ys = [], []
        # for i in xrange(opt['n']):

        #     xc,yc =   Variable(x.cuda(model.gidxs[i]), volatile=True), \
        #             Variable(y.squeeze().cuda(model.gidxs[i]), volatile=True)
        #     xs.append(xc)
        #     ys.append(yc)

        # fs, errs = model(xs, ys)
        # fs = [fs[i].data[0] for i in xrange(opt['n'])]

        # f.update(np.mean(fs), bsz)
        # top1.update(np.mean(errs), bsz)
        xc,yc = Variable(x.cuda(gpus[0]), volatile=True), \
                Variable(y.squeeze().cuda(gpus[0]), volatile=True)

        yh = model.reference(xc)
        _f = nn.CrossEntropyLoss()(yh, yc).data[0]
        prec1, = accuracy(yh.data, yc.data, topk=(1,))
        err = 100. - prec1[0]
        f.update(_f, bsz)
        top1.update(err, bsz)

    print((color('red', '**[%2d] %2.4f %2.4f%%\n'))%(e, f.avg, top1.avg))
    print('')

for e in xrange(opt['e'], opt['B']):
    train(e)
    val(e)