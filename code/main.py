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

# if opt['n'] > 1:
#     opt['nw'] = 0

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
                            'f','v', 'augment', 't', 'nw',
                            'save','e','l2','r', 'lr'])
logger = create_logger(opt)
pprint(opt)

dataset, augment = getattr(loader, opt['dataset'])(opt)
loaders = loader.get_loaders(dataset, augment, opt)

params = dict(t=0, gdot=opt['gdot']/len(loaders[0]['train']))
opt.update(**params)
optimizer = getattr(optim, opt['optim'])(model, config=opt)

def train(e):
    optimizer.config['lr'] = lrschedule(opt, e, logger)
    model.train()

    n = opt['n']
    ids = deepcopy(model.ids)

    meters = AverageMeters(['f', 'top1', 'top5', 'dt'])

    bsz = opt['b']
    maxb = int(len(loaders[0]['train_full'])*opt['frac'])
    iters = [loaders[i]['train'].__iter__() for i in xrange(n)]

    for bi in xrange(maxb):
        _dt = timer()
        def helper():
            def feval():
                xs, ys = [None]*n, [None]*n
                fs, errs, errs5 = [0]*n, [0]*n, [0]*n

                for i in xrange(n):
                    try:
                        x, y = next(iters[i])
                    except StopIteration:
                        if iters[i].num_workers > 0:
                            iters[i]._shutdown_workers()
                        time.sleep(0.1)
                        iters[i] = loaders[i]['train'].__iter__()
                        x, y = next(iters[i])
                    xs[i], ys[i] =  Variable(x.cuda(ids[i])), Variable(y.squeeze().cuda(ids[i]))

                yhs = model(xs, ys)
                for i in xrange(n):
                    fs[i] = criterion.cuda(ids[i])(yhs[i], ys[i])
                    errs[i], errs5[i] = clerr(yhs[i].data, ys[i].data, topk=(1,5))
                model.backward(fs)

                fs = [fs[i].data[0] for i in xrange(n)]
                return fs, errs, errs5
            return feval

        fs, errs, errs5 = optimizer.step(helper())
        _dt = timer() - _dt
        meters.add(dict(f=np.mean(fs), top1=np.mean(errs), top5=np.mean(errs5), dt=_dt))

        mm = meters.value()
        if opt['l'] and bi % 25 ==0 and bi > 0:
            s = dict(i=bi + e*maxb, e=e, train=True)
            s.update(**mm)
            logger.info('[LOG] ' + json.dumps(s))

        bif = int(5/mm['dt'])+1
        if bi % bif == 0 and bi > 0:
            print((color('blue', '[%2.2fs][%2d][%4d/%4d] %2.4f %2.2f%% %2.2f%%'))%(_dt,
                e,bi,maxb, mm['f'], mm['top1'], mm['top5']))

    mm = meters.value()
    if opt['l']:
        s = dict(e=e, i=0, train=True)
        s.update(**mm)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('blue', '++[%2d] %2.4f %2.2f%% %2.2f%% [%2.2fs]'))% (e, mm['f'], mm['top1'], mm['top5'], meters.m['dt'].sum))
    print()

def val(e):
    n = opt['n']
    ids = deepcopy(model.ids)

    rid = model.refid
    val_model = model.w[0] if n == 1 else model.ref

    if (not 'imagenet' in opt['dataset']):
        dry_feed(val_model, loaders[0]['train_full'], mid=rid)

    model.eval()
    meters = AverageMeters(['f', 'top1', 'top5'])

    for bi, (x,y) in enumerate(loaders[0]['val']):
        bsz = x.size(0)

        xc,yc = Variable(x.cuda(rid), volatile=True), \
                Variable(y.squeeze().cuda(rid), volatile=True)

        yh = val_model(xc)
        f = criterion.cuda(rid)(yh, yc).data[0]
        err, err5 = clerr(yh.data, yc.data, topk=(1,5))
        meters.add(dict(f=f, top1=err, top5=err5))

        mm = meters.value()
        if bi % 100 == 0 and bi > 0:
            print((color('red', '*[%d][%2d] %2.4f %2.4f%% %2.4f%%'))%(e, bi, \
                    mm['f'], mm['top1'], mm['top5']))

    mm = meters.value()
    if opt['l']:
        s = dict(e=e, i=0, value=True)
        s.update(**mm)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('red', '**[%2d] %2.4f %2.4f%% %2.4f%%\n'))%(e, mm['f'], mm['top1'], mm['top5']))
    print('')

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
    print('[Loaded model, check validation error]')
    val(d['e'])

    opt['e'] = d['e'] + 1

    print('[Loading new optimizer]')
    params = dict(t=d['t'], gdot=opt['gdot']/len(loaders[0]['train']))
    optimizer = getattr(optim, opt['optim'])(model, config = opt.update(**params))

for e in xrange(opt['e'], opt['B']):
    train(e)
    val(e)
    save_ensemble(e)