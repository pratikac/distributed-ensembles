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

class member_t():
    def __init__(self, gidx):
        self.gidx = gidx
        self.set_gpu()

        self.model = getattr(models, opt['m'])(opt)
        self.train_loader, self.val_loader, \
            self.test_loader = getattr(loader, opt['dataset'])(opt)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = getattr(optim, opt['optim'])(self.model.parameters(),
            config = dict(lr=opt['lr'], momentum=0.9, nesterov=True, weight_decay=opt['l2'],
            L=opt['L'], eps=opt['eps'], g0=opt['g0'], g1=opt['g1'], verbose=opt['v']))

        self.model.cuda()
        self.criterion.cuda()

    def set_gpu(self):
        th.cuda.set_device(self.gidx)

ensemble = [member_t(gpus[i%len(gpus)]) for i in xrange(opt['n'])]

def train(e):
    def schedule(optimizer, e):
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

    def train_helper(model, criterion, optimizer, train_loader, e):
        schedule(optimizer, e)

        model.train()

        fs, top1 = AverageMeter(), AverageMeter()
        ts = timer()

        bsz = opt['b']
        maxb = len(train_loader)

        for bi in xrange(maxb):
            def helper():
                def feval(bprop=True):
                    x,y = next(train_loader)
                    x, y = Variable(x.cuda()), Variable(y.squeeze().cuda())
                    bsz = x.size(0)

                    optimizer.zero_grad()
                    yh = model(x)
                    f = criterion.forward(yh, y)
                    if bprop:
                        f.backward()

                    prec1, = accuracy(yh.data, y.data, topk=(1,))
                    err = 100.-prec1[0]
                    return (f.data[0], err)
                return feval

            f, err = optimizer.step(helper(), model, criterion)
            th.cuda.synchronize()

            fs.update(f, bsz)
            top1.update(err, bsz)

            # if opt['l']:
            #     s = dict(i=bi + e*maxb, e=e, f=f, top1=err)
            #     logger.info('[LOG] ' + json.dumps(s))

            if bi % 25 == 0 and bi != 0:
                print((color('blue', '[%2d][%4d/%4d] %2.4f %2.2f%%'))%(e,bi,maxb,
                    fs.avg, top1.avg))

        # if opt['l']:
        #     s = dict(e=e, i=0, f=fs.avg, top1=top1.avg, train=True)
        #     logger.info('[SUMMARY] ' + json.dumps(s))
        #     logger.info('')

        print(  (color('blue', '++[%2d] %2.4f %2.2f%% [%.2fs]'))% (e,
                fs.avg, top1.avg, timer()-ts))

    i = 0
    for r in ensemble:
        print('Replica %d'%(i))
        i += 1
        th.cuda.set_device(r.gidx)
        train_helper(r.model, r.criterion, r.optimizer, r.train_loader, e)

def val(e):
    def set_dropout(model, cache = None, p=0):
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

    def dry_feed(model, train_loader):
        model.train()
        cache = set_dropout()
        maxb = len(train_loader)
        for bi in xrange(maxb):
            x,y = next(train_loader)
            x,y =   Variable(x.cuda(), volatile=True), \
                    Variable(y.squeeze().cuda(), volatile=True)
            yh = model(x)
        set_dropout(cache)

    def val_helper(model, criterion, train_loader, val_loader):
        dry_feed(model, train_loader)
        model.eval()

        maxb = len(val_loader)
        fs, top1 = AverageMeter(), AverageMeter()
        for bi in xrange(maxb):
            x,y = next(val_loader)
            bsz = x.size(0)

            x,y =   Variable(x.cuda(), volatile=True), \
                    Variable(y.squeeze().cuda(), volatile=True)
            yh = model(x)

            f = criterion.forward(yh, y).data[0]
            prec1, = accuracy(yh.data, y.data, topk=(1,))
            err = 100-prec1[0]

            fs.update(f, bsz)
            top1.update(err, bsz)

        # if opt['l']:
        #     s = dict(e=e, i=0, f=fs.avg, top1=top1.avg, val=True)
        #     logger.info('[SUMMARY] ' + json.dumps(s))
        #     logger.info('')

        print((color('red', '**[%2d] %2.4f %2.4f%%\n'))%(e, fs.avg, top1.avg))
        print('')

    i = 0
    for r in ensemble:
        print('Replica %d'%(i))
        i += 1
        th.cuda.set_device(r.gidx)
        val_helper(r.model, r.criterion, r.train_loader, r.val_loader)

# def validate_ensemble(ensemble, data_loader):
#     for m in ensemble:
#         dry_feed(m)
#         m.eval()

#     maxb = len(data_loader)
#     fs, top1 = AverageMeter(), AverageMeter()
#     for bi in xrange(maxb):
#         x,y = next(data_loader)
#         bsz = x.size(0)

#         x,y =   Variable(x.cuda(), volatile=True), \
#                 Variable(y.squeeze().cuda(), volatile=True)

#         yh = F.softmax(ensemble[0](x))
#         for m in ensemble[1:]:
#             yh += F.softmax(m(x))
#         yh = yh/float(len(ensemble))

#         #f = criterion.forward(yh, y).data[0]
#         f = 0
#         prec1, = accuracy(yh.data, y.data, topk=(1,))
#         err = 100-prec1[0]

#         fs.update(f, bsz)
#         top1.update(err, bsz)

#     print((color('red', 'Ensemble pred: ** %2.4f %2.4f%%\n'))%(fs.avg, top1.avg))
#     print('')

# if opt['validate_ensemble'] != '':
#     ensemble = []
#     for f in sorted(glob.glob(opt['validate_ensemble'] + '*.pz')):
#         m = deepcopy(model)
#         print('Loading ' + f)
#         d = th.load(f)
#         m.load_state_dict(d['state_dict'])
#         m = m.cuda()
#         ensemble.append(m)

#     #print('Train')
#     #validate_ensemble(ensemble, train_loader)
#     print('Val')
#     validate_ensemble(ensemble, val_loader)

# for e in xrange(opt['e'], opt['B']):
#     train(e)
#     if e % opt['f'] == opt['f'] -1:
#         val(e, val_loader)
#     if opt['save']:
#         save(model, opt, marker='s_%s'%opt['s'])
# else:
#     e = 0
#     val(e, test_loader)
#     print('Training error')
#     val(e, train_loader)

for e in xrange(opt['e'], opt['B']):
    train(e)
    if e % opt['f'] == opt['f'] -1:
        val(e)