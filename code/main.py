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
['--optim', 'SGD', 'ESGD | SGD'],
['--dataset', 'mnist', 'mnist | rotmnist | cifar10 | cifar100'],
['--frac', 1.0, 'fraction of dataset'],
['-b', 128, 'batch_size'],
['--augment', False, 'data augmentation'],
['-e', 0, 'start epoch'],
['-B', 100, 'Max epochs'],
['--depth', 28, 'wrn depth'],
['--widen', 10, 'wrn widen'],
['--lr', 0.1, 'learning rate'],
['--lrs', '', 'learning rate schedule'],
['--l2', 0.0, 'ell-2'],
['-d', 0.0, 'dropout'],
['-L', 0, 'sgld iterations'],
['--eps', 1e-4, 'sgld noise'],
['--g0', 1e-4, 'gamma'],
['--g1', 0.0, 'scoping'],
['-s', 42, 'seed'],
['-g', 2, 'gpu idx'],
['-l', False, 'log'],
['-f', 10, 'print freq'],
['-v', False, 'verbose'],
['--retrain', '', 'checkpoint'],
['--validate', '', 'validate a checkpoint'],
['--validate_ensemble', '', 'validate an ensemble'],
['--ensemble_mean', '', 'mean of ensemble'],
['--ensemble_std', '', 'std of ensemble'],
['--save', False, 'save network']
])
if opt['L'] > 0:
    opt['f'] = 1
if opt['l']:
    opt['f'] = 1

th.set_num_threads(2)
if opt['g'] in [0, 1, 2]:
    th.cuda.set_device(opt['g'])
random.seed(opt['s'])
np.random.seed(opt['s'])
th.manual_seed(opt['s'])
th.cuda.manual_seed_all(opt['s'])
cudnn.benchmark = True

model = getattr(models, opt['m'])(opt)
train_loader, val_loader, test_loader,_ = getattr(loader, opt['dataset'])(opt)
if opt['g'] > 2:
    model = th.nn.DataParallel(model)
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = getattr(optim, opt['optim'])(model.parameters(),
        config = dict(lr=opt['lr'], momentum=0.9, nesterov=False, weight_decay=opt['l2'],
        L=opt['L'], eps=opt['eps'], g0=opt['g0'], g1=opt['g1'], verbose=opt['v']))

ckpt = None
if not opt['retrain'] == '':
    ckpt = th.load(opt['retrain'])
if not opt['validate'] == '':
    ckpt = th.load(opt['validate'])

if ckpt is not None:
    model.load_state_dict(ckpt['state_dict'])
    print('Loading model: %s'%ckpt['name'])

build_filename(opt, blacklist=['lrs','retrain','step', \
                            'ratio','f','v','dataset', 'augment', 'd',
                            'depth', 'widen','save','e','validate','l2','eps',
                            'ensemble_mean','ensemble_std','validate_ensemble'])
logger = create_logger(opt)
pprint(opt)

def schedule(e):
    if opt['lrs'] == '':
        opt['lrs'] = json.dumps([[opt['B'], opt['lr']]])

    lrs = json.loads(opt['lrs'])

    idx = len(lrs)-1
    for i in xrange(len(lrs)):
        if e < lrs[i][0]:
            idx = i
            break
    lr = lrs[idx][1]

    print('[LR]: ', lr)
    if opt['l']:
        logger.info('[LR] ' + json.dumps({'lr': lr}))
    optimizer.config['lr'] = lr

def train(e):
    schedule(e)
    loader = train_loader.__iter__()

    model.train()

    fs, top1 = AverageMeter(), AverageMeter()
    ts = timer()

    bsz = opt['b']
    maxb = len(loader)

    for bi in xrange(maxb):
        def helper():
            def feval(bprop=True):
                x,y = next(loader)
                x, y = Variable(x.cuda(async=True)), Variable(y.squeeze().cuda(async=True))
                bsz = x.size(0)

                model.zero_grad()
                yh = model(x)
                f = criterion.forward(yh, y)

                if bprop:
                    f.backward()

                err = 100. - accuracy(yh.data, y.data, topk=(1,))
                return (f.data[0], err)
            return feval

        f, err = optimizer.step(helper(), model, criterion)
        th.cuda.synchronize()

        fs.update(f, bsz)
        top1.update(err, bsz)

        if opt['l']:
            s = dict(i=bi + e*maxb, e=e, f=f, top1=err)
            logger.info('[LOG] ' + json.dumps(s))

        bif = opt['L'] > 0 and 5 or 25
        if bi % bif == 0 and bi != 0:
            print((color('blue', '[%2d][%4d/%4d] %2.4f %2.2f%%'))%(e,bi,maxb,
                fs.avg, top1.avg))

    if opt['l']:
        s = dict(e=e, i=0, f=fs.avg, top1=top1.avg, train=True)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print(  (color('blue', '++[%2d] %2.4f %2.2f%% [%.2fs]'))% (e,
            fs.avg, top1.avg, timer()-ts))

def set_dropout(cache = None, p=0):
    if cache is None:
        cache = []
        for l in model.modules():
            if 'Dropout' in str(type(l)):
                cache.append(l.p)
                l.p = p
        return cache
    else:
        for l in model.modules():
            if 'Dropout' in str(type(l)):
                assert len(cache) > 0, 'cache is empty'
                l.p = cache.pop(0)

def dry_feed(model):
    model.train()
    cache = set_dropout()
    maxb = len(train_loader)
    for bi in xrange(maxb):
        x,y = next(train_loader)
        x,y =   Variable(x.cuda(async=True), volatile=True), \
                Variable(y.squeeze().cuda(async=True), volatile=True)
        yh = model(x)
    set_dropout(cache)

def val(e, data_loader):
    dry_feed(model)
    model.eval()

    maxb = len(data_loader)
    fs, top1 = AverageMeter(), AverageMeter()
    for bi in xrange(maxb):
        x,y = next(data_loader)
        bsz = x.size(0)

        x,y =   Variable(x.cuda(async=True), volatile=True), \
                Variable(y.squeeze().cuda(async=True), volatile=True)
        yh = model(x)

        f = criterion.forward(yh, y).data[0]
        err = 100. - accuracy(yh.data, y.data, topk=(1,))

        fs.update(f, bsz)
        top1.update(err, bsz)

    if opt['l']:
        s = dict(e=e, i=0, f=fs.avg, top1=top1.avg, val=True)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('red', '**[%2d] %2.4f %2.4f%%\n'))%(e, fs.avg, top1.avg))
    print('')


def validate_ensemble(ensemble, data_loader):
    for m in ensemble:
        dry_feed(m)
        m.eval()

    maxb = len(data_loader)
    fs, top1 = AverageMeter(), AverageMeter()
    for bi in xrange(maxb):
        x,y = next(data_loader)
        bsz = x.size(0)

        x,y =   Variable(x.cuda(), volatile=True), \
                Variable(y.squeeze().cuda(), volatile=True)

        yh = F.softmax(ensemble[0](x))
        for m in ensemble[1:]:
            yh += F.softmax(m(x))
        yh = yh/float(len(ensemble))

        #f = criterion.forward(yh, y).data[0]
        f = 0
        err = 100 - accuracy(yh.data, y.data, topk=(1,))

        fs.update(f, bsz)
        top1.update(err, bsz)

    print((color('red', 'Ensemble pred: ** %2.4f %2.4f%%\n'))%(fs.avg, top1.avg))
    print('')

if opt['ensemble_mean'] != '' and opt['ensemble_std'] != '':
    mu = OrderedDict(json.load(open(opt['ensemble_mean'], 'rb')))
    std = OrderedDict(json.load(open(opt['ensemble_std'], 'rb')))
    for k in mu:
        mu[k] = np.array(mu[k], dtype=np.float32)
    for k in std:
        std[k] = np.array(std[k], dtype=np.float32)

    for i in xrange(10):
        d = deepcopy(mu)
        for k in d:
            d[k] = th.from_numpy(d[k] + 0.5*std[k]*np.random.randn(*std[k].shape))

        model.load_state_dict(d)
        model = model.cuda()
        print('Created model %d'%i)
        print('Train')
        val(0, train_loader)
        print('Val')
        val(0, test_loader)

if opt['validate_ensemble'] != '':
    ensemble = []
    for f in sorted(glob.glob(opt['validate_ensemble'] + '*.pz')):
        m = deepcopy(model)
        print('Loading ' + f)
        d = th.load(f)
        m.load_state_dict(d['state_dict'])
        m = m.cuda()
        ensemble.append(m)

    #print('Train')
    #validate_ensemble(ensemble, train_loader)
    print('Val')
    validate_ensemble(ensemble, val_loader)

if opt['validate'] == '':
    for e in xrange(opt['e'], opt['B']):
        train(e)
        if e % opt['f'] == opt['f'] -1:
            val(e, val_loader)
        if opt['save']:
            save(model, opt, marker='s_%s'%opt['s'])
else:
    e = 0
    val(e, test_loader)
    print('Training error')
    val(e, train_loader)