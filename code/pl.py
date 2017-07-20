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
['-m', 'lenets', 'lenet'],
['--dataset', 'mnist', 'mnist'],
['-g', 0, 'gpu idx'],
['-n', 1, 'num loaders'],
['--gpus', '', 'gpus'],
['-d', 0.0, 'dropout'],
['-b', 128, 'batch_size'],
['-e', 100, 'epochs'],
['-B', 100, 'max epochs'],
['--lr', 0.1, 'learning rate'],
['--lrs', '[[30,0.1],[60,0.01],[90,0.001],[100,0.0001]]', 'lrs'],
['-s', 42, 'seed'],
['--nw', 0, 'workers'],
['--augment', False, 'augment'],
['--frac', 1.0, 'frac'],
['-l', False, 'log'],
['-v', False, 'verbose'],
])

gid = opt['g']

ngpus = th.cuda.device_count()
gpus = [i if opt['g'] >= ngpus else opt['g'] for i in xrange(ngpus)]
if not opt['gpus'] == '':
    gpus = json.loads(opt['gpus'])
setup(t=4, s=opt['s'], gpus=gpus)

dataset, augment = getattr(loader, opt['dataset'])(opt)
loaders = loader.get_loaders(dataset, augment, opt)
mnist = loaders[0]['train_full']

opt['b'] = 16384
loaders_lbsz = loader.get_loaders(dataset, augment, opt)
mnist_lbsz = loaders_lbsz[0]['train_full']
opt['b'] = 128

model = models.lenets(opt).cuda(gid)
criterion = nn.CrossEntropyLoss().cuda(gid)
optimizer = th.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)

build_filename(opt, blacklist=['lrs', 'optim', 'gpus', 'gdot', 'depth', 'widen',
                            'f','v', 'augment', 't', 'nw',
                            'save','e','l2','r', 'lr'])
logger = create_logger(opt)
pprint(opt)

N = models.num_parameters(model)
fw, dfw = th.FloatTensor(N).cuda(gid), th.FloatTensor(N).cuda(gid)
optim.flatten_params(model, fw, dfw)

def full_grad():
    grad = th.FloatTensor(N).cuda(gid).zero_()
    loss = 0

    for bi, (x,y) in enumerate(mnist_lbsz):
        xc,yc = Variable(x.cuda(gid)), Variable(y.squeeze().cuda(gid))
        model.zero_grad()
        yh = model(xc)
        f = criterion(yh, yc)
        f.backward()

        loss += f.data[0]
        grad.add_(dfw)

    loss /= float(len(mnist_lbsz))
    grad /= float(len(mnist_lbsz))
    return loss, grad

for e in xrange(opt['e']):
    model.train()

    lr = lrschedule(opt, e, logger)
    for g in optimizer.param_groups:
        g['lr'] = lr

    maxb = len(mnist)
    for bi, (x,y) in enumerate(mnist):
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
        ff, fgrad = full_grad()
        s['fullf'] = ff
        s['fulldw'] = fgrad.norm()
        s['dw_fulldw'] = dfw.dot(fgrad)/dfw.norm()/fgrad.norm()

        if bi % 25 == 0 and bi > 0 and opt['v']:
            print s, timer()-dt

        if opt['l']:
            logger.info('[LOG] ' + json.dumps(s))
