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
                            'f','v','dataset', 'augment', 'd',
                            'depth', 'widen','save','e','validate','l2','eps',
                            'validate_ensemble'])
logger = create_logger(opt)
pprint(opt)

model = models.ReplicateModel(getattr(models, opt['m'])(opt), \
            nn.CrossEntropyLoss(), opt['n'], gpus)
train_loader, val_loader, test_loader = getattr(loader, opt['dataset'])(opt)


def train(e):
    model.train()

    bsz = opt['b']
    maxb = len(train_loader)

    for bi in xrange(maxb):
        def helper():
            def feval(bprop=True):
                xs, ys = [], []
                for i in xrange(opt['n']):
                    x, y =  next(train_loader)
                    x, y =  Variable(x.cuda(model.gidxs[i])), \
                            Variable(y.squeeze().cuda(model.gidxs[i]))
                    xs.append(x)
                    ys.append(y)

                model.zero_grad()
                fs, errs = model(xs, ys)
                model.backward()

                if bi % 100 == 0:
                    fs = [fs[i].data[0] for i in xrange(opt['n'])]
                    print(fs, errs)

                return fs, errs
            return feval

        feval = helper()
        feval()

def val(e):
    pass

for e in xrange(opt['e'], opt['B']):
    train(e)
    if e % opt['f'] == opt['f'] -1:
        val(e)