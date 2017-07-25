from __future__ import print_function
import argparse, math, random
import torch as th
import torch.nn as nn
import torchnet as tnt

from torch.autograd import Variable

from exptutils import *
import models, loader, optim
from timeit import default_timer as timer

import numpy as np
import logging
from pprint import pprint
import pdb, glob, sys, gc, time
from copy import deepcopy

opt = add_args([
['-o', '/local2/pratikac/results', 'output'],
['-m', 'lenet', 'lenet | mnistfc | allcnn | wrn* | resnet*'],
['--dataset', 'mnist', 'mnist | cifar10 | cifar100 | svhn | imagenet'],
['-g', 3, 'gpu idx'],
['--gpus', '', 'groups of gpus'],
['--frac', 1.0, 'fraction of dataset'],
['-b', 128, 'batch_size'],
['--augment', False, 'data augmentation'],
['-e', 0, 'start epoch'],
['--optim', 'ProxSGD', 'Parle | SGD | EntropySGD | ProxSGD'],
['-d', -1., 'dropout'],
['--l2', -1., 'ell-2'],
['-B', 100, 'max epochs'],
['--lr', 0.1, 'learning rate'],
['--lrs', '', 'learning rate schedule'],
['-n', 1, 'num replicas'],
['--ni', 1, 'num replicas to execute simultaneously'],
['-L', 25, 'sgld iterations'],
['--g0', 0.01, 'SGLD gamma'],
['--g1', 1.0, 'elastic gamma'],
['--gdot', 0.5, 'gamma dot'],
['-s', 42, 'seed'],
['--nw', 4, 'workers'],
['-l', False, 'log'],
['-f', 10, 'print freq'],
['-v', False, 'verbose'],
['-r', '', 'resume ckpt'],
['--save', False, 'save ckpt'],
])

if opt['L'] > 0 or opt['l']:
    opt['f'] = 1

ngpus = th.cuda.device_count()
gpus = [i if opt['g'] >= ngpus else opt['g'] for i in xrange(ngpus)]
if not opt['gpus'] == '':
    gpus = json.loads(opt['gpus'])
setup(t=4, s=opt['s'], gpus=gpus)

model = models.FederatedModel(opt, gpus=gpus)
criterion = nn.CrossEntropyLoss()
dataset, augment = getattr(loader, opt['dataset'])(opt)
loaders = loader.get_federated_loaders(dataset, augment, opt)

build_filename(opt, blacklist=['lrs', 'optim', 'gpus', 'gdot', 'depth', 'widen',
                            'f','v', 'augment', 't', 'nw',
                            'save','e','l2','r', 'lr'])
logger = create_logger(opt)
pprint(opt)

for e in xrange(1):
    n = opt['n']

    x = th.randn(opt['b'], 1, 28, 28)
    y = next(loaders[0]['val'].__iter__())[1]

    gids = deepcopy(model.ids)

    for b in xrange(10):
        ns = range(opt['n'])
        ni = opt['ni']
        for ids in [ns[i:i+ni] for i in xrange(0, opt['n'], ni)]:

            xs, ys, fs = [], [], []
            for ii in ids:
                t1, t2 = Variable(x.cuda(gids[ii])), Variable(y.cuda(gids[ii]))
                xs.append(t1)
                ys.append(t2)

            yhs = model.forward(ids, xs, ys)
            for iii, ii in enumerate(ids):
                fs.append(criterion.cuda(gids[ii])(yhs[iii], ys[iii]))
            model.backward(ids, fs)

            print(ids)
