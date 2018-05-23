import torch as th
import torch.distributed as dist
import numpy as np

import torch.nn as nn
import torchnet as tnt
from torchvision.datasets.mnist import MNIST
from torch.autograd import Variable

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from timeit import default_timer as timer

import sys, argparse, random, pdb, os
from copy import deepcopy
import pprint
pp = pprint.PrettyPrinter(indent=4)

from exptutils import *
import models, loader
from timeit import default_timer as timer
import logging

opt = add_args([
['-o', '/home/%s/local2/pratikac/results'%os.environ['USER'], 'output'],
['-m', 'lenet', 'lenet | mnistfc | allcnn | wrn* | resnet*'],
['--dataset', 'mnist', 'mnist | cifar10 | cifar100 | svhn | imagenet'],
['-g', 3, 'gpu idx'],
['-n', 1, 'replicas'],
['-r', 0, 'rank'],
['--gpus', '', 'groups of gpus'],
['--frac', 1.0, 'fraction of dataset'],
['-b', 128, 'batch_size'],
['--bbsz', 0, 'very large batch sizes'],
['--augment', True, 'data augmentation'],
['-e', 0, 'start epoch'],
['-d', -1., 'dropout'],
['--l2', -1., 'ell-2'],
['-B', 5, 'max epochs'],
['--lr', 0.1, 'learning rate'],
['--lrs', '', 'learning rate schedule'],
['--Ls', '', 'schedule for Langevin steps'],
['--burnin', 1, 'burnin for small batches'],
['-L', 25, 'sgld iterations'],
['--gamma', 0.01, 'gamma'],
['--rho', 0.01, 'rho'],
['-s', 42, 'seed'],
['--nw', 0, 'workers'],
['-l', False, 'log'],
['-f', 10, 'print freq'],
['-v', False, 'verbose'],
['--resume', '', 'resume ckpt'],
['--save', False, 'save best ckpt'],
['--save_all', False, 'save all ckpt'],
])

ngpus = th.cuda.device_count()
gpus = [i if opt['g'] >= ngpus else opt['g'] for i in range(ngpus)]
if not opt['gpus'] == '':
    gpus = json.loads(opt['gpus'])
setup(s=opt['s']+opt['r'])
opt['g'] = gpus[int(opt['r'] % len(gpus))]
th.cuda.set_device(opt['g'])

if opt['n'] > 1:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('nccl', rank=opt['r'], world_size=opt['n'])

if not opt['dataset'] == 'imagenet':
    dataset, augment = getattr(loader, opt['dataset'])(opt)
    ds_bsz = loader.get_dataset(dataset, augment, opt)
else:
    ds_bsz = getattr(loader, opt['dataset'])(opt)
ds_bsz = ds_bsz[opt['r']]

bsz = opt['b']
opt['b'] = 128
if not opt['dataset'] == 'imagenet':
    dataset, augment = getattr(loader, opt['dataset'])(opt)
    ds_128 = loader.get_dataset(dataset, augment, opt)
else:
    ds_128 = getattr(loader, opt['dataset'])(opt)
ds_128 = ds_128[opt['r']]
opt['b'] = bsz

model = getattr(models, opt['m'])(opt, microbn=True).cuda()
criterion = nn.CrossEntropyLoss().cuda()
if opt['bbsz'] > 0:
    bbsz_model = models.BBszModel(opt, model, criterion, mbsz=opt['bbsz'])

build_filename(opt, blacklist=['lrs', 'optim', 'gpus', 'gdot', 'depth', 'widen',
                            'f','v', 'augment', 't', 'nw', 'save_all', 'd',
                            'save','e','l2','r', 'lr', 'Ls', 'b', 'resume', 'frac', 'burnin', 'bbsz', 'L'])
logger = create_logger(opt)
if opt['r'] == 0:
    pp.pprint(opt)

def parle_step(sync=False):

    mom, alpha = 0.9, 0.0
    lr = opt['lr']
    r = opt['r']
    nb = opt['nb']

    if not 'state' in opt:
        opt['state'] = {}
        s = opt['state']
        s['t'] = 0

        for k in ['za', 'muy', 'mux', 'xa', 'x', 'cache']:
            s[k] = {}

        for p in model.parameters():
            for k in ['za', 'muy', 'mux', 'xa']:
                s[k][p] = p.data.clone()

            s['muy'][p].zero_()
            s['mux'][p].zero_()

            s['x'][p] = p.data.clone()
            s['cache'][p] = p.data.clone()

    s = opt['state']
    t = s['t']

    za, muy, mux, xa, x, cache = s['za'], s['muy'], s['mux'], \
        s['xa'], s['x'], s['cache']

    gamma = opt['gamma']*(1 + 0.5/nb)**t
    rho = opt['rho']*(1 + 0.5/nb)**t
    gamma, rho = min(gamma, 1), min(rho, 1)

    # dst lives on the master
    def sync_ref(src, dst):
        if opt['n'] > 1:
            for p in model.parameters():
                s['cache'][p].copy_(src[p])
                dist.reduce(s['cache'][p], dst=0, op=dist.reduce_op.SUM)

            if r == 0:
                for p in model.parameters():
                    dst[p] = s['cache'][p]/float(opt['n'])

            for p in model.parameters():
                dist.broadcast(dst[p], src=0)
        else:
            for p in model.parameters():
                s['cache'][p].copy_(src[p])
                dst[p].copy_(src[p])

    if sync:
        # add another sync, helps with large L
        sync_ref(za, x)

        for p in model.parameters():
            # elastic-sgd term
            p.grad.data.zero_()
            p.grad.data.add_(1.0, xa[p] - za[p]).add_(rho*opt['L'], xa[p] - x[p])

            mux[p].mul_(mom).add_(p.grad.data)
            p.grad.data.add_(mux[p])
            p.data.add_(-lr, p.grad.data)

            xa[p].copy_(p.data)

        sync_ref(xa, x)
        s['t'] += 1

    else:
        # entropy-sgd iterations
        for p in model.parameters():
            p.grad.data.add_(gamma, p.data - xa[p])

            muy[p].mul_(mom).add_(p.grad.data)
            p.grad.data.add_(muy[p])
            p.data.add_(-lr, p.grad.data)

            za[p].mul_(alpha).add_(1-alpha, p.data)

# @do_profile(follow=[parle_step])
def train(e):
    opt['lr'] = lrschedule(opt, e, logger)
    opt['L'] = Lschedule(opt, e, logger)

    model.train()

    ds = ds_bsz
    if e < opt['burnin']:
        opt['lr'] = opt['lr']*((e+1)/float(opt['burnin']))

    opt['nb'] = int(len(ds['train_full'])*opt['frac'])
    train_iter = ds['train'].__iter__()

    meters = AverageMeters(['f', 'top1', 'top5', 'dt'])
    buf = dict( f=th.FloatTensor(1).zero_().cuda(),
                top1=th.FloatTensor(1).zero_().cuda(),
                top5=th.FloatTensor(1).zero_().cuda())

    for b in range(opt['nb']):

        loss, top1, top5 = None, None, None
        _dt = timer()
        for l in range(opt['L']):
            try:
                x,y = next(train_iter)
            except StopIteration:
                train_iter = ds['train'].__iter__()
                x,y = next(train_iter)

            x, y = Variable(x).cuda(async=True), Variable(y).cuda(async=True)

            if opt['bbsz'] > 0:
                f, yh = bbsz_model.forward_backward(x, y)
            else:
                model.zero_grad()
                yh = model(x)
                f = criterion(yh, y)
                f.backward()

                if opt['l2'] > 0:
                    for p in model.parameters():
                        p.grad.data.add_(opt['l2'], p.data)

            if l == 0:
                top1, top5 = clerr(yh.data, y.data, (1,5))
                loss = f.item()
                buf['f'][0], buf['top1'][0], buf['top5'][0] = loss, top1, top5

            parle_step()

        parle_step(sync=True)
        _dt = timer() - _dt

        for k in buf:
            dist.reduce(buf[k], dst=0, op=dist.reduce_op.SUM)
            buf[k] /= float(opt['n'])

        meters.add(dict(f=buf['f'].item(), top1=buf['top1'].item(), top5=buf['top5'].item(), dt=_dt))

        mm = meters.value()
        if opt['l'] and b % 25 == 0 and b > 0:
            if opt['r'] == 0:
                s = dict(i=b + e*opt['nb'], e=e, train=True)
                s.update(**mm)
                logger.info('[LOG] ' + json.dumps(s))

        bif = int(5/mm['dt'])+1
        if b % bif == 0 and b > 0:
            print((color('blue', '[%2.2fs][%2d][%4d/%4d] %2.4f %2.2f%% %2.2f%%'))%(_dt,
                e,b,opt['nb'], mm['f'], mm['top1'], mm['top5']))

    mm = meters.value()
    if opt['l'] and opt['r'] == 0:
        s = dict(e=e, i=0, train=True)
        s.update(**mm)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('blue', '++[%2d] %2.4f %2.2f%% %2.2f%% [%2.2fs]'))% (e, mm['f'], mm['top1'], mm['top5'], meters.m['dt'].sum))
    print()
    return mm

def dry_feed(m):
    def set_dropout(cache = None, p=0):
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

    m.train()
    cache = set_dropout()
    with th.no_grad():
        for _, (x,y) in enumerate(ds_128['train_full']):
            x = Variable(x).cuda(async=True)
            m(x)
    set_dropout(cache)

def validate(e):
    m = deepcopy(model)
    for p,q in zip(m.parameters(), model.parameters()):
        p.data.copy_(opt['state']['x'][q])

    dry_feed(m)
    m.eval()

    meters = AverageMeters(['f', 'top1', 'top5'])

    for b, (x,y) in enumerate(ds_bsz['val']):
        x, y = Variable(x).cuda(async=True), Variable(y).cuda(async=True)

        yh = m(x)
        f = criterion(yh, y).item()

        top1, top5 = clerr(yh.data, y.data, (1,5))
        meters.add(dict(f=f, top1=top1, top5=top5))

        mm = meters.value()
        if b % 100 == 0 and b > 0:
            print((color('red', '*[%d][%2d] %2.4f %2.4f%% %2.4f%%'))%(e, b, \
                    mm['f'], mm['top1'], mm['top5']))

    mm = meters.value()
    if opt['l']:
        s = dict(e=e, i=0, val=True)
        s.update(**mm)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('red', '**[%2d] %2.4f %2.4f%% %2.4f%%\n'))%(e, mm['f'], mm['top1'], mm['top5']))
    print('')
    return mm

for e in range(opt['B']):
    if opt['r'] == 0:
        print()

    r = train(e)

    if opt['r'] == 0:
        validate(e)