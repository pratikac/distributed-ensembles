import torch as th
import torch.distributed as dist
import numpy as np

import torch.nn as nn
import torchnet as tnt
from torchvision.datasets.mnist import MNIST
from torch.autograd import Variable

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

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
['--augment', True, 'data augmentation'],
['-e', 0, 'start epoch'],
['-d', -1., 'dropout'],
['--l2', -1., 'ell-2'],
['-B', 5, 'max epochs'],
['--lr', 0.1, 'learning rate'],
['--lrs', '', 'learning rate schedule'],
['--Ls', '', 'schedule for Langevin steps'],
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
setup(t=4, s=opt['s'])
opt['g'] = gpus[int(opt['r'] % len(gpus))]
th.cuda.set_device(opt['g'])

build_filename(opt, blacklist=['lrs', 'optim', 'gpus', 'gdot', 'depth', 'widen',
                            'f','v', 'augment', 't', 'nw', 'save_all', 'd',
                            'save','e','l2','r', 'lr', 'Ls', 'b', 'gamma', 'rho'])
logger = create_logger(opt)
if opt['r'] == 0:
    pp.pprint(opt)

# normalize rho
opt['rho'] = opt['rho']*opt['L']*opt['n']

if opt['n'] > 1:
    # initialize distributed comm
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=opt['r'], world_size=opt['n'])

if not opt['dataset'] == 'imagenet':
    dataset, augment = getattr(loader, opt['dataset'])(opt)
    loaders = loader.get_loaders(dataset, augment, opt)
else:
    loaders = getattr(loader, opt['dataset'])(opt)

loader = loaders[opt['r']]

model = getattr(models, opt['m'])(opt).cuda()
criterion = nn.CrossEntropyLoss().cuda()

def parle_step(sync=False):
    eps = 1e-3

    mom, alpha = 0.9, 0.75
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

    gamma = opt['gamma']*(1 + 0.5/nb)**(t // opt['L'])
    rho = opt['rho']*(1 + 0.5/nb)**(t // opt['L'])
    gamma, rho = min(gamma, 1), min(rho, 10)

    def sync_with_master(xa, x):
        if opt['n'] > 1:
            for p in model.parameters():
                s['cache'][p].copy_(xa[p])
                dist.all_reduce(s['cache'][p], op=dist.reduce_op.SUM)

            if r == 0:
                for p in model.parameters():
                    x[p] = s['cache'][p]/float(opt['n'])

            for p in model.parameters():
                dist.broadcast(x[p], src=0)
        else:
            for p in model.parameters():
                s['cache'][p].copy_(xa[p])
                x[p].copy_(xa[p])

    if sync:
        # add another sync, helps with large L
        sync_with_master(za, x)

        for p in model.parameters():

            # elastic-sgd term
            p.grad.data.zero_()
            p.grad.data.add_(1, xa[p] - za[p]).add_(rho, xa[p] - x[p])

            mux[p].mul_(mom).add_(p.grad.data)
            p.grad.data.add_(mux[p])
            p.data.add_(-lr, p.grad.data)

            xa[p].copy_(p.data)
        sync_with_master(xa, x)
    else:
        # entropy-sgd iterations
        for p in model.parameters():
            p.grad.data.add_(gamma, p.data - xa[p])

            muy[p].mul_(mom).add_(p.grad.data)
            p.grad.data.add_(muy[p])
            p.data.add_(-lr, p.grad.data)

            za[p].mul_(alpha).add_(1-alpha, p.data)

    s['t'] += 1


from line_profiler import LineProfiler
def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    return inner


# @do_profile(follow=[parle_step])
def train(e):
    opt['lr'] = lrschedule(opt, e, logger)
    opt['L'] = Lschedule(opt, e, logger)

    model.train()

    opt['nb'] = int(len(loader['train_full'])*opt['frac'])
    train_iter = loader['train'].__iter__()

    loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter()

    for b in range(opt['nb']):
        for l in range(opt['L']):
            try:
                x,y = next(train_iter)
            except StopIteration:
                train_iter = loader['train'].__iter__()
                x,y = next(train_iter)

            x, y = Variable(x).cuda(async=True), Variable(y).cuda(async=True)

            model.zero_grad()
            yh = model(x)
            f = criterion(yh, y)
            f.backward()

            if opt['l2'] > 0:
                for p in model.parameters():
                    p.grad.data.add_(opt['l2'], p.data)

            if l == 0:
                top1.add(yh.data, y.data)
                loss.add(f.item())

                if b % 100 == 0 and b > 0:
                    print('[%03d][%03d/%03d] %.3f %.3f%%'%(e, b, opt['nb'], \
                            loss.value()[0], top1.value()[0]))

            parle_step()

        parle_step(sync=True)

    r = dict(f=loss.value()[0], top1=top1.value()[0])
    print('+[%02d] %.3f %.3f%%'%(e, r['f'], r['top1']))
    return r

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
        for _, (x,y) in enumerate(loader['train_full']):
            x = Variable(x).cuda(async=True)
            m(x)
    set_dropout(cache)

def validate(e):
    m = deepcopy(model)
    for p,q in zip(m.parameters(), model.parameters()):
        p.data.copy_(opt['state']['x'][q])

    dry_feed(m)
    m.eval()

    loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter()

    for b, (x,y) in enumerate(loader['val']):
        x, y = Variable(x).cuda(async=True), Variable(y).cuda(async=True)

        yh = m(x)
        f = criterion(yh, y)

        top1.add(yh.data, y.data)
        loss.add(f.item())

    r = dict(f=loss.value()[0], top1=top1.value()[0])
    print('*[%02d] %.3f %.3f%%'%(e, r['f'], r['top1']))
    return r

for e in range(opt['B']):
    if opt['r'] == 0:
        print()

    r = train(e)

    if opt['r'] == 0:
        validate(e)

    opt['lr'] /= 10.0
