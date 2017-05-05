from __future__ import print_function
import argparse, math, random
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import torch.backends.cudnn as cudnn

from exptutils import *
import models, loader, optim
import numpy as np
import logging
from pprint import pprint

opt = add_args([
['-o', '/local2/pratikac/results', 'output'],
['-m', 'ptbs', 'ptbs | ptbl'],
['--hdim', 200, 'hdim'],
['-d', 0.2, 'dropout'],
['--optim', 'ESGD', 'ESGD | HJB | PME | FB | LL | PMEAVG | SGLD | SGD'],
['-b', 20, 'batch_size'],
['-e', 0, 'start epoch'],
['-T', 35, 'bptt'],
['-B', 40, 'Max epochs'],
['--lr', 20, 'learning rate'],
['--l2', 0.0, 'ell-2'],
['--clip', 0.25, 'gradient clipping'],
['--lrs', '', 'learning rate schedule'],
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

corpus, ptb, loader = loader.ptb(opt)
opt['vocab'] = len(corpus.dictionary)
model = getattr(models, opt['m'])(opt)
if opt['g'] > 2:
    model = th.nn.DataParallel(model)
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = getattr(optim, opt['optim'])(model.parameters(),
        config = dict(lr=opt['lr'], momentum=0.0, nesterov=True, weight_decay=opt['l2'],
        L=opt['L'], eps=opt['eps'], g0=opt['g0'], g1=opt['g1'], verbose=opt['v']))

ckpt = None
if not opt['retrain'] == '':
    ckpt = th.load(opt['retrain'])
if not opt['validate'] == '':
    ckpt = th.load(opt['retrain'])

if ckpt is not None:
    model.load_state_dict(ckpt['state_dict'])
    print('Loading model: %s'%ckpt['name'])

build_filename(opt, blacklist=['lrs','retrain', 'f','v', \
                            'save','e','validate','eps','T',
                            'vocab', 'l2'])
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
    #schedule(e)

    model.train()

    fs,perp = AverageMeter(), AverageMeter()
    ts = timer()

    bsz = opt['b']
    maxb = (ptb['train'].size(0) -1) // opt['T']

    h = model.init_hidden(opt['b'])
    for bi in xrange(maxb):
        def helper():
            def feval(bprop=True):
                idx = int(np.random.random()*maxb)*opt['T']

                x,y = loader(ptb['train'], idx)
                x, y = Variable(x.cuda()), Variable(y.squeeze().cuda())
                bsz = x.size(0)

                _h = models.repackage_hidden(h)

                model.zero_grad()
                yh, hh = model(x, _h)
                f = criterion(yh.view(-1, opt['vocab']), y)
                if bprop:
                    f.backward()

                nn.utils.clip_grad_norm(model.parameters(), opt['clip'])

                f = f.data[0]
                return (f, math.exp(f))
            return feval

        f, p = optimizer.step(helper(), model, criterion)
        th.cuda.synchronize()

        fs.update(f, bsz)
        perp.update(p, bsz)

        if opt['l']:
            s = dict(i=bi + e*maxb, e=e, f=f, perp=p)
            logger.info('[LOG] ' + json.dumps(s))

        if bi % 100 == 0 and bi != 0:
            print((color('blue', '[%2d][%4d/%4d] %2.4f %2.4f'))%(e,bi,maxb,
                fs.avg, perp.avg))

    if opt['l']:
        s = dict(e=e, i=0, f=fs.avg, perp=perp.avg, train=True)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print(  (color('blue', '++[%2d] %2.4f %2.4f [%.2fs]'))% (e,
            fs.avg, perp.avg, timer()-ts))

def val(e, src):
    model.eval()

    h = model.init_hidden(opt['b'])
    fs,perp = AverageMeter(), AverageMeter()

    for bi in xrange(0, ptb[src].size(0)-1, opt['T']):
        h = models.repackage_hidden(h)

        x,y = loader(ptb[src], bi)
        bsz = x.size(0)

        x,y =   Variable(x.cuda(), volatile=True), \
                Variable(y.squeeze().cuda(), volatile=True)
        yh, hh = model(x, h)

        f = criterion(yh.view(-1, opt['vocab']), y).data[0]

        fs.update(f, bsz)
        perp.update(math.exp(f), bsz)

    if opt['l']:
        s = dict(e=e, i=0, f=fs.avg, perp=perp.avg, val=True)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('red', '**[%2d] %2.4f %2.4f\n'))%(e, fs.avg, perp.avg))
    print('')

for e in xrange(opt['e'], opt['B']):
    train(e)
    if e % opt['f'] == opt['f'] -1:
        val(e, 'valid')
    if opt['save']:
        save(model, opt, marker='e_%s'%e)
val(opt['B']-1, 'test')