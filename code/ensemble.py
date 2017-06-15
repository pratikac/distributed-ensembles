from __future__ import print_function
import argparse, math, random
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
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
['--dataset', 'mnist', 'mnist | cifar10 | cifar100'],
['-g', 3, 'gpu idx'],
['--gpus', '', 'groups of gpus'],
['--frac', 1.0, 'fraction of dataset'],
['-b', 128, 'batch_size'],
['--augment', False, 'data augmentation'],
['-e', 0, 'start epoch'],
['--optim', 'DistESGD', 'optim'],
['-d', -1., 'dropout'],
['--l2', -1., 'ell-2'],
['-B', 100, 'Max epochs'],
['--lr', 0.1, 'learning rate'],
['--lrd', 0., 'lrd'],
['--lrs', '', 'learning rate schedule'],
['-n', 1, 'replicas'],
['-L', 25, 'sgld iterations'],
['--g0', 0.01, 'SGLD gamma'],
['--g1', 1.0, 'elastic gamma'],
['--gdot', 1e-3, 'gamma dot'],
['-s', 42, 'seed'],
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
setup(  t=4, s=opt['s'],
        gpus=gpus)
if opt['dataset'] == 'imagenet':
    opt['b'] = 32

model = models.ReplicateModel(opt, gpus=gpus)
criterion = nn.CrossEntropyLoss()

build_filename(opt, blacklist=['lrs', 'optim', 'lrd', 'gpus',
                            'f','v', 'augment', 't',
                            'save','e','l2','r', 'lr'])
logger = create_logger(opt)
pprint(opt)

loaders = []
for i in xrange(opt['n']):
    tr,v,te,trf = getattr(loader, opt['dataset'])(opt)
    loaders.append(dict(train=tr,val=v,test=te,train_full=trf))

optimizer = getattr(optim, opt['optim'])(model, config =
        dict(lr=opt['lr'], lrd=opt['lrd'], weight_decay=opt['l2'], L=opt['L'],
            g0 = opt['g0'], g1 = opt['g1'], gdot=opt['gdot'],
            verbose=opt['v'],
            t=0))

def train(e):
    optimizer.config['lr'] = lrschedule(opt, e, logger)
    model.train()

    f, top1, top5, dt = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    fstd, top1std, top5std = AverageMeter(), AverageMeter(), AverageMeter()

    bsz = opt['b']
    maxb = len(loaders[0]['train'])
    t0 = timer()

    n = opt['n']
    ids = deepcopy(model.ids)

    for bi in xrange(maxb):
        _dt = timer()
        def helper():
            def feval():
                xs, ys = [None]*n, [None]*n
                fs, errs, errs5 = [None]*n, [None]*n, [None]*n

                for i in xrange(n):
                    x, y = next(loaders[i]['train'])
                    xs[i], ys[i] =  Variable(x.cuda(ids[i], async=True)), \
                            Variable(y.squeeze().cuda(ids[i], async=True))

                yhs = model(xs, ys)
                for i in xrange(n):
                    fs[i] = criterion.cuda(ids[i])(yhs[i], ys[i])
                    acc = accuracy(yhs[i].data, ys[i].data, topk=(1,5))
                    errs[i] = 100. - acc[0]
                    errs5[i] = 100. - acc[1]
                model.backward(fs)

                fs = [fs[i].data[0] for i in xrange(n)]
                return fs, errs, errs5
            return feval

        fs, errs, errs5 = optimizer.step(helper())

        f.update(np.mean(fs), bsz)
        fstd.update(np.std(fs), bsz)

        top1.update(np.mean(errs), bsz)
        top1std.update(np.std(errs), bsz)
        top5.update(np.mean(errs5), bsz)
        top5std.update(np.std(errs5), bsz)
        dt.update(timer()-_dt, 1)

        if opt['l']:
            s = dict(i=bi + e*maxb, e=e, f=np.mean(fs), top1=np.mean(errs), top5=np.mean(errs5),
                    fstd=np.std(fs), top1std=np.std(errs), top5std = np.std(errs5), dt=dt.avg)
            logger.info('[LOG] ' + json.dumps(s))

        bif = int(5/dt.avg)+1
        if bi % bif == 0 and bi != 0:
            print((color('blue', '[%2.2fs][%2d][%4d/%4d] %2.4f+-%2.4f %2.2f+-%2.2f%% %2.2f+-%2.2f%%'))%(dt.avg,
                e,bi,maxb, f.avg, fstd.avg, top1.avg, top1std.avg, top5.avg, top5std.avg))

    if opt['l']:
        s = dict(e=e, i=0, f=f.avg, fstd=fstd.avg, top1=top1.avg, top1std=top1std.avg,
                top5=top5.avg, top5std=top5std.avg,
                train=True, t=timer()-t0)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('blue', '++[%2d] %2.4f+-%2.4f %2.2f+-%2.2f%% %2.2f+-%2.2f%% [%2.2fs]'))% (e,
        f.avg, fstd.avg, top1.avg, top1std.avg, top5.avg, top5std.avg, timer()-t0))
    print()

def val(e):
    n = opt['n']
    ids = deepcopy(model.ids)

    if opt['frac'] < 1:
        model.train()
        print((color('red', 'Full train:')))
        for i in xrange(n):
            maxb = len(loaders[i]['train_full'])
            f, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
            for bi in xrange(maxb):
                x,y = next(loaders[i]['train_full'])
                bsz = x.size(0)

                x,y = Variable(x.cuda(ids[i], async=True), volatile=True), \
                    Variable(y.squeeze().cuda(ids[i], async=True), volatile=True)

                yh = model.w[i](x)
                _f = criterion.cuda(ids[i])(yh, y).data[0]
                acc = accuracy(yh.data, y.data, topk=(1,5))
                err, err5 = 100. - acc[0], 100. - acc[1]
                f.update(_f, bsz)
                top1.update(err, bsz)
                top5.update(err5, bsz)
            print((color('red', '++[%d][%2d] %2.4f %2.4f%% %2.4f%%'))%(e, i, f.avg, top1.avg, top5.avg))

    rid = model.refid
    if not opt['m'][:6] == 'resnet':
        dry_feed(model.ref, loaders[0]['train_full'], id=rid)
    model.eval()
    val_loader = loaders[0]['val']
    maxb = len(val_loader)
    f, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    for bi in xrange(maxb):
        x,y = next(val_loader)
        bsz = x.size(0)

        xc,yc = Variable(x.cuda(rid, async=True), volatile=True), \
                Variable(y.squeeze().cuda(rid, async=True), volatile=True)

        yh = model.ref(xc)
        _f = criterion.cuda(rid)(yh, yc).data[0]
        acc = accuracy(yh.data, yc.data, topk=(1,5))
        err, err5 = 100. - acc[0], 100. - acc[1]
        f.update(_f, bsz)
        top1.update(err, bsz)
        top5.update(err5, bsz)
        print((color('red', '++[%d][%2d] %2.4f %2.4f%% %2.4f%%'))%(e, bi, f.avg, top1.avg, top5.avg))

    if opt['l']:
        s = dict(e=e, i=0, f=f.avg, top1=top1.avg, top5=top5.avg, val=True)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('red', '**[%2d] %2.4f %2.4f%% %2.4f%%\n'))%(e, f.avg, top1.avg, top5.avg))
    print('')

def save_ensemble(e):
    if not opt['save']:
        return

    loc = opt.get('o','/local2/pratikac/results')
    th.save(dict(
            ref=model.ref.state_dict(),
            w = [model.w[i].state_dict() for i in xrange(opt['n'])],
            e=e,
            t=optimizer.state['t']),
            os.path.join(loc, opt['filename']+'.pz'))

if not opt['r'] == '':
    print('Loading model from: ', opt['r'])
    d = th.load(opt['r'])
    model.ref.load_state_dict(d['ref'])
    model.ref = model.ref.cuda(model.refid)
    for i in xrange(opt['n']):
        model.w[i].load_state_dict(d['w'][i])
        model.w[i] = model.w[i].cuda(model.ids[i])
    opt['e'] = d['e'] + 1

    print('Loading new optimizer')
    optimizer = optim.DistESGD(config =
        dict(lr=opt['lr'], weight_decay=opt['l2'], L=opt['L'],
            g0 = opt['g0'], g1 = opt['g1'],
            verbose=opt['v'],
            t=d['t']))

    print('Loaded model, validation')
    val(opt['e'])

for e in xrange(opt['e'], opt['B']):
    train(e)
    val(e)
    save_ensemble(e)
