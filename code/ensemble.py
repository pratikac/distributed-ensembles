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
['--optim', 'DistESGD', 'optim: DistESGD | SGD | ElasticSGD | EntropySGD'],
['-d', -1., 'dropout'],
['--l2', -1., 'ell-2'],
['-B', 100, 'Max epochs'],
['--lr', 0.1, 'learning rate'],
['--lrs', '', 'learning rate schedule'],
['-n', 1, 'replicas'],
['-L', 25, 'sgld iterations'],
['--g0', 0.01, 'SGLD gamma'],
['--g1', 1.0, 'elastic gamma'],
['--gdot', 0.5, 'gamma dot'],
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
setup(t=4, s=opt['s'], gpus=gpus)

model = models.ReplicateModel(opt, gpus=gpus)
criterion = nn.CrossEntropyLoss()

build_filename(opt, blacklist=['lrs', 'optim', 'gpus', 'gdot', 'depth', 'widen',
                            'f','v', 'augment', 't',
                            'save','e','l2','r', 'lr'])
logger = create_logger(opt)
pprint(opt)

loaders = []
if opt['frac'] > 1-1e-12:
    tr,v,_,_ = getattr(loader, opt['dataset'])(opt)
    for i in xrange(opt['n']):
        loaders.append(dict(train=tr,val=v,test=tr,train_full=tr))
else:
    for i in xrange(opt['n']):
        opt['frac_start'] = (i/float(opt['n'])) % 1
        tr,v,te,trf = getattr(loader, opt['dataset'])(opt)
        loaders.append(dict(train=tr,val=v,test=te,train_full=trf))

train_iters = [None]*opt['n']

optimizer = getattr(optim, opt['optim'])(model, config =
        dict(lr=opt['lr'], weight_decay=opt['l2'], L=opt['L'], llr=lrschedule(opt, opt['e']),
            g0 = opt['g0'], g1 = opt['g1'], gdot=opt['gdot']/len(loaders[0]['train']),
            verbose=opt['v'],
            t=0))

def train(e):
    optimizer.config['lr'] = lrschedule(opt, e, logger)
    model.train()

    n = opt['n']
    ids = deepcopy(model.ids)

    f, top1, top5, dt = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    fstd, top1std, top5std = AverageMeter(), AverageMeter(), AverageMeter()
    pf, ptop1, ptop5 = [],[],[]

    bsz = opt['b']
    maxb = len(loaders[0]['train'])
    for i in xrange(n):
        train_iters[i] = loaders[i]['train'].__iter__()

    t0 = timer()

    for bi in xrange(maxb):
        _dt = timer()
        def helper():
            def feval():
                xs, ys = [None]*n, [None]*n
                fs, errs, errs5 = [0]*n, [0]*n, [0]*n

                for i in xrange(n):
                    if 'threaded' in opt['dataset']:
                        try:
                            x, y = next(train_iters[i])
                        except StopIteration:
                            train_iters[i] = loaders[i]['train'].__iter__()
                            x, y = next(train_iters[i])
                    else:
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

        pf.append(fs)
        ptop1.append(errs)
        ptop5.append(errs5)

        if opt['l'] and bi % 25 ==0 and bi > 0:
            s = dict(i=bi + e*maxb, e=e, f=np.mean(fs), top1=np.mean(errs), top5=np.mean(errs5),
                    fstd=np.std(fs), top1std=np.std(errs), top5std = np.std(errs5), dt=timer() - _dt)
            logger.info('[LOG] ' + json.dumps(s))

        bif = int(5/dt.avg)+1
        if bi % bif == 0 and bi > 0:
            print((color('blue', '[%2.2fs][%2d][%4d/%4d] %2.4f+-%2.4f %2.2f+-%2.2f%% %2.2f+-%2.2f%%'))%(timer() - _dt,
                e,bi,maxb, np.mean(pf), np.mean(np.std(pf, 1)), np.mean(ptop1), np.mean(np.std(ptop1,1)),
                np.mean(ptop5), np.mean(np.std(ptop5,1)) ))
            pf, ptop1, ptop5 = [],[],[]

    if opt['l']:
        s = dict(e=e, i=0, f=f.avg, fstd=fstd.avg, top1=top1.avg, top1std=top1std.avg,
                top5=top5.avg, top5std=top5std.avg,
                train=True, t=timer()-t0)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('blue', '++[%2d] %2.4f+-%2.4f %2.2f+-%2.2f%% %2.2f+-%2.2f%% [%2.2fs]'))% (e,
        f.avg, fstd.avg, top1.avg, top1std.avg, top5.avg, top5std.avg, timer()-t0))
    print()

def check_models(m1, m2):
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert (p1-p2).norm() < 1e-12

def val(e):
    n = opt['n']
    ids = deepcopy(model.ids)

    if opt['frac'] < 1 and False and (not 'imagenet' in opt['dataset']):
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
    val_model = model.ref
    if n == 1:
        val_model = model.w[0]
    if (not 'imagenet' in opt['dataset']):
        dry_feed(val_model, loaders[0]['train_full_iter'], mid=rid)
    model.eval()

    valiter = loaders[0]['val']
    if 'threaded' in opt['dataset']:
        valiter = loaders[0]['val'].__iter__()
    maxb = len(valiter)

    f, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    for bi in xrange(maxb):
        x,y = next(valiter)
        bsz = x.size(0)

        xc,yc = Variable(x.cuda(rid), volatile=True), \
                Variable(y.squeeze().cuda(rid), volatile=True)

        yh = val_model(xc)
        _f = criterion.cuda(rid)(yh, yc).data[0]
        acc = accuracy(yh.data, yc.data, topk=(1,5))
        err, err5 = 100. - acc[0], 100. - acc[1]
        f.update(_f, bsz)
        top1.update(err, bsz)
        top5.update(err5, bsz)

        if bi % 100 == 0 and bi > 0:
            print((color('red', '*[%d][%2d] %2.4f %2.4f%% %2.4f%%'))%(e, bi, f.avg, top1.avg, top5.avg))

    if opt['l']:
        s = dict(e=e, i=0, f=f.avg, top1=top1.avg, top5=top5.avg, val=True)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('red', '**[%2d] %2.4f %2.4f%% %2.4f%%\n'))%(e, f.avg, top1.avg, top5.avg))
    print('')

    del valiter

def save_ensemble(e):
    if not opt['save']:
        return

    loc = opt.get('o','/local2/pratikac/results')
    dirloc = os.path.join(loc, opt['m'], opt['filename'])
    if not os.path.isdir(dirloc):
        os.makedirs(dirloc)

    r = gitrev(opt)
    meta = dict(SHA=r[0], STATUS=r[1], DIFF=r[2])
    th.save(dict(
            meta = meta,
            opt=json.dumps(opt),
            ref=model.ref.state_dict(),
            w = [model.w[i].state_dict() for i in xrange(opt['n'])],
            e=e,
            t=optimizer.state['t']),
            os.path.join(dirloc, str(e) + '.pz'))

if not opt['r'] == '':
    print('Loading model from: ', opt['r'])
    d = th.load(opt['r'])
    model.ref.load_state_dict(d['ref'])
    model.ref = model.ref.cuda(model.refid)
    for i in xrange(opt['n']):
        model.w[i].load_state_dict(d['w'][i])
        model.w[i] = model.w[i].cuda(model.ids[i])
    opt['e'] = d['e'] + 1

    print('[Loading new optimizer]')
    optimizer = getattr(optim, opt['optim'])(model, config =
        dict(lr=opt['lr'], weight_decay=opt['l2'], L=opt['L'], llr=lrschedule(opt, opt['e']),
            g0 = opt['g0'], g1 = opt['g1'], gdot=opt['gdot']/len(loaders[0]['train']),
            verbose=opt['v'],
            t=d['t']))

    print('[Loaded model, check validation error]')
    val(opt['e'])

for e in xrange(opt['e'], opt['B']):
    train(e)
    val(e)
    save_ensemble(e)
