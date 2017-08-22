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

opt = add_args([
['-o', '/local2/pratikac/results', 'output'],
['-m', 'lenet', 'lenet | allcnns'],
['--dataset', 'mnist', 'mnist | cifar10'],
['-g', 0, 'gpu idx'],
['-n', 1, 'num loaders'],
['--gpus', '', 'gpus'],
['-d', 0.0, 'dropout'],
['-b', 128, 'batch_size'],
['--bb', 128, 'batch_size'],
['--full_grad', False, 'calculate full grad'],
['-B', 10, 'max epochs'],
['--lr', 0.1, 'learning rate'],
['--l2', -1.0, 'ell2'],
['--lrs', '[[6,0.1],[10,0.01]]', 'lrs'],
['-s', 42, 'seed'],
['--nw', 0, 'workers'],
['--augment', False, 'augment'],
['--frac', 1.0, 'frac'],
['-l', False, 'log'],
['-v', False, 'verbose'],
])

gid = opt['g']
ngpus = th.cuda.device_count()
if gid >= ngpus:
    gid = 0

gpus = [i if opt['g'] >= ngpus else opt['g'] for i in xrange(ngpus)]
if not opt['gpus'] == '':
    gpus = json.loads(opt['gpus'])
setup(t=4, s=opt['s'], gpus=gpus)

dataset, augment = getattr(loader, opt['dataset'])(opt)
loaders = loader.get_loaders(dataset, augment, opt)
data = loaders[0]['train_full']

b = opt['b']
opt['b'] = opt['bb']
loaders_lbsz = loader.get_loaders(dataset, augment, opt)
data_lbsz = loaders_lbsz[0]['train_full']
opt['b'] = b

model = getattr(models, opt['m'])(opt).cuda(gid)
if opt['g'] >= ngpus:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss().cuda(gid)
optimizer = th.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)

# populate buffers before flattening params
for bi, (x,y) in enumerate(data):
    xc,yc = Variable(x.cuda(gid)), Variable(y.squeeze().cuda(gid))
    model.zero_grad()
    yh = model(xc)
    f = criterion(yh, yc)
    f.backward()
    break

build_filename(opt, blacklist=['lrs', 'optim', 'gpus', 'gdot', 'depth', 'widen',
                            'f','v', 'augment', 't', 'nw', 'frac', 'nw', 'frac', 'd', 'b',
                            'save','e','l2','r', 'lr', 'bb', 'full_grad'])
logger = create_logger(opt)
pprint(opt)

N = models.num_parameters(model)
fw, dfw = th.FloatTensor(N).cuda(gid), th.FloatTensor(N).cuda(gid)
optim.flatten_params(model, fw, dfw)

def full_grad():
    grad = th.FloatTensor(N).cuda(gid).zero_()
    loss, top1, top5 = 0, 0, 0
    n = float(len(data_lbsz))

    for bi, (x,y) in enumerate(data_lbsz):
        xc,yc = Variable(x.cuda(gid)), Variable(y.squeeze().cuda(gid))
        model.zero_grad()
        yh = model(xc)
        f = criterion(yh, yc)
        f.backward()

        err, err5 = clerr(yh.data, yc.data, topk=(1,5))

        loss += f.data[0]
        grad.add_(dfw)
        top1 += err
        top5 += err5

    loss /= n
    grad /= n
    top1 /= n
    top5 /= n
    return loss, grad, top1, top5

for e in xrange(opt['B']):
    model.train()

    lr = lrschedule(opt, e, logger)
    for g in optimizer.param_groups:
        g['lr'] = lr

    maxb = len(data)
    for bi, (x,y) in enumerate(data):
        bsz = x.size(0)
        dt = timer()

        xc,yc = Variable(x.cuda(gid)), Variable(y.squeeze().cuda(gid))

        fwc = fw.clone()

        model.zero_grad()
        yh = model(xc)
        f = criterion(yh, yc)
        f.backward()

        optimizer.step()

        err, err5 = clerr(yh.data, yc.data, topk=(1,5))
        s = dict(i=bi + e*maxb, e=e,
                f=f.data[0], top1=err, top5=err5,
                dw=dfw.norm(), w=fw.norm(),
                deltaw=(fw-fwc).norm(),
                dt=timer()-dt)

        if opt['full_grad']:
            ff, fgrad, ftop1, ftop5 = full_grad()
            s['fullf'] = ff
            s['ftop1'], s['ftop5'] = ftop1, ftop5
            s['fulldw'] = fgrad.norm()
            s['vardw'] = (fgrad - dfw).norm()**2
            s['biasdw'] = (fgrad - dfw).sum()
            s['pl'] = fgrad.norm()**2/2./ff
            s['dw_fulldw'] = dfw.dot(fgrad)/dfw.norm()/fgrad.norm()

        bif = 1 if opt['full_grad'] and opt['v'] else 25
        if bi % bif == 0 and bi > 0:
            print s, timer()-dt

        if opt['l']:
            logger.info('[LOG] ' + json.dumps(s))
