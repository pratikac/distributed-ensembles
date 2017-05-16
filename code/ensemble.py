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
['--optim', 'DistributedESGD', 'DistributedESGD'],
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
['-n', 1, 'replicas'],
['-L', 0, 'sgld iterations'],
['--eps', 1e-4, 'sgld noise'],
['--g00', 0.03, 'gamma Langevin'],
['--g01', 0.0, 'scoping Langevin'],
['--g10', 0.03, 'gamma elastic'],
['--g11', 0.0, 'scoping elastic'],
['--alpha', 0.0, 'alpha, loss: f + alpha fkld'],
['--b0', 1.0, 'beta, dw = grad f + (1-b0)*w + g*b0*(w-mu)'],
['--beta', 0.5, 'temperature in dark knowledge'],
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
th.set_num_threads(opt['t'])
random.seed(opt['s'])
np.random.seed(opt['s'])
th.manual_seed(opt['s'])
th.cuda.manual_seed_all(opt['s'])
cudnn.benchmark = True

build_filename(opt, blacklist=['lrs','retrain','step', \
                            'f','v','dataset', 'augment', 'd', 't',
                            'depth', 'widen','save','e','validate','l2','eps',
                            'validate_ensemble', 'alpha'])
logger = create_logger(opt)
pprint(opt)

model = models.ReplicateModel(opt, \
        nn.CrossEntropyLoss(), nn.KLDivLoss(), gpus)

train_loaders = []
val_loader, test_loader = None, None
train_loader_full = None
for i in xrange(opt['n']):
    tl, val_loader, test_loader,train_loader_full = getattr(loader, opt['dataset'])(opt)
    train_loaders.append(tl)

optimizer = getattr(optim, opt['optim'])(config =
        dict(lr=opt['lr'], momentum=0.9, nesterov=True, weight_decay=opt['l2'],
            L=opt['L'], eps=opt['eps'],
            g00=opt['g00'], g01=opt['g01'],
            g10=opt['g10'], g11=opt['g11'],
            verbose=opt['v']
        ))

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
    model.train()

    f, top1, dt = AverageMeter(), AverageMeter(), AverageMeter()
    fstd, top1std = AverageMeter(), AverageMeter()

    bsz = opt['b']
    maxb = len(train_loaders[0])
    t0 = timer()

    # xs = [Variable(th.randn(opt['b'],3,32,32).cuda(model.gidxs[i])) for i in xrange(opt['n'])]
    # ys = [Variable((th.rand(opt['b'],)*10).long().cuda(model.gidxs[i])) for i in xrange(opt['n'])]
    #fs, errs = [None for i in xrange(opt['n'])], [None for i in xrange(opt['n'])]

    fs = [0 for i in xrange(opt['n'])]
    errs = [0 for i in xrange(opt['n'])]

    for bi in xrange(maxb):
        _dt = timer()

        def helper():
            def feval():
                xs, ys, yhs =   [None for i in xrange(opt['n'])], \
                                [None for i in xrange(opt['n'])], \
                                [None for i in xrange(opt['n'])]

                # if opt['alpha'] > 0:
                #     x, y = next(train_loaders[0])
                for i in xrange(opt['n']):
                    if opt['alpha'] < 1e-6:
                        x, y = next(train_loaders[i])
                    xs[i], ys[i] =  Variable(x.cuda(model.gidxs[i], async=True)), \
                            Variable(y.squeeze().cuda(model.gidxs[i], async=True))

                fs, errs = model(xs, ys)
                model.backward()
                fs = [fs[i].data[0] for i in xrange(opt['n'])]
                return fs, errs
            return feval

        fs, errs = optimizer.step(helper(), model)

        f.update(np.mean(fs), bsz)
        fstd.update(np.std(fs), bsz)

        top1.update(np.mean(errs), bsz)
        top1std.update(np.std(errs), bsz)
        dt.update(timer()-_dt, 1)

        if opt['l']:
            s = dict(i=bi + e*maxb, e=e, f=np.mean(fs), top1=np.mean(errs), fstd=np.std(fs), top1std=np.std(errs))
            logger.info('[LOG] ' + json.dumps(s))

        if bi % 25 == 0:
            print((color('blue', '[%2.2fs][%2d][%4d/%4d] %2.4f+-%2.4f %2.2f+-%2.2f%%'))%(dt.avg, e,bi,maxb,
                f.avg, fstd.avg, top1.avg, top1std.avg))

    if opt['l']:
        s = dict(e=e, i=0, f=f.avg, top1=top1.avg, train=True)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

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
        maxb = len(train_loader_full)
        for bi in xrange(maxb):
            x,y = next(train_loader_full)
            x,y =   Variable(x.cuda(0, async=True), volatile=True), \
                    Variable(y.squeeze().cuda(0, async=True), volatile=True)
            yh = m(x)
        set_dropout(m,cache)

    print((color('red', 'Full train:')))
    for i in xrange(opt['n']):
        maxb = len(train_loader_full)
        f, top1 = AverageMeter(), AverageMeter()
        for bi in xrange(maxb):
            x,y = next(train_loader_full)
            bsz = x.size(0)

            x,y = Variable(x.cuda(gpus[model.gidxs[i]], async=True), volatile=True), \
                Variable(y.squeeze().cuda(gpus[model.gidxs[i]], async=True), volatile=True)

            yh = model.ensemble[i](x)
            _f = nn.CrossEntropyLoss()(yh, y).data[0]
            prec1, = accuracy(yh.data, y.data, topk=(1,))
            err = 100. - prec1[0]
            f.update(_f, bsz)
            top1.update(err, bsz)
        print((color('red', '++[%d][%2d] %2.4f %2.4f%%'))%(e, i, f.avg, top1.avg))

    dry_feed(model.reference)
    model.eval()

    print((color('red', 'Full val:')))
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
        xc,yc = Variable(x.cuda(gpus[0], async=True), volatile=True), \
                Variable(y.squeeze().cuda(gpus[0], async=True), volatile=True)

        yh = model.reference(xc)
        _f = nn.CrossEntropyLoss()(yh, yc).data[0]
        prec1, = accuracy(yh.data, yc.data, topk=(1,))
        err = 100. - prec1[0]
        f.update(_f, bsz)
        top1.update(err, bsz)

    if opt['l']:
        s = dict(e=e, i=0, f=f.avg, top1=top1.avg, val=True)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('red', '**[%2d] %2.4f %2.4f%%\n'))%(e, f.avg, top1.avg))
    print('')

def save_ensemble():
    if not opt['save']:
        return

    loc = opt.get('o','/local2/pratikac/results')
    for i in xrange(len(model.ensemble)):
        d = model.ensemble[i].state_dict()
        dr = []
        for k in d:
            dr.append(d[k].cpu().numpy().tolist())
        json.dump(dr, open(os.path.join(loc, opt['m']+'_'+str(i)+'.json'), 'wb'))

for e in xrange(opt['e'], opt['B']):
    train(e)
    val(e)
    if e % 50 == 0 and e > 0:
        save_ensemble()