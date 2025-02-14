from __future__ import print_function
import argparse, math, random
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import torch.nn.functional as F
from torch.nn.parallel import scatter, parallel_apply, gather

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
['-m', 'ptbs', 'ptbs | ptbl'],
['--dataset', 'ptb', 'ptb'],
['-g', 0, 'gpu idx'],
['--gpus', '', 'groups of gpus'],
['-b', 20, 'batch_size'],
['-e', 0, 'start epoch'],
['--optim', 'DistESGD', 'optim: DistESGD | SGD | HJ | ElasticSGD | EntropySGD'],
['--l2', -1., 'ell-2'],
['-B', 40, 'Max epochs'],
['-T', 35, 'bptt'],
['--lr', 20.0, 'learning rate'],
['--llr', 20.0, 'llr'],
['--lrs', '', 'learning rate schedule'],
['--mom', 0.5, 'mom'],
['--clip', 0.25, 'gradient clipping'],
['-n', 1, 'replicas'],
['-L', 5, 'sgld iterations'],
['--g0', 0.01, 'SGLD gamma'],
['--g1', 1.0, 'elastic gamma'],
['--gdot', 0.5, 'gamma dot'],
['--beta1', 0.75, 'beta1'],
['-s', 42, 'seed'],
['-l', False, 'log'],
['-f', 10, 'print freq'],
['-v', False, 'verbose'],
['-r', '', 'resume ckpt'],
['--save', False, 'save ckpt'],
])

if opt['n'] > 1:
    opt['g'] = th.cuda.device_count()

if opt['L'] > 0 or opt['l']:
    opt['f'] = 1

ngpus = th.cuda.device_count()
gpus = [i if opt['g'] >= ngpus else opt['g'] for i in xrange(ngpus)]
if not opt['gpus'] == '':
    gpus = json.loads(opt['gpus'])
setup(  t=4, s=opt['s'],
        gpus=gpus)

corpus, ptb, batcher = None, [None]*opt['n'], [None]*opt['n']
for i in xrange(opt['n']):
    corpus, ptb[i], batcher[i] = loader.ptb(opt)
opt['vocab'] = len(corpus.dictionary)

model = models.ReplicateModel(opt, gpus=gpus)
criterion = nn.CrossEntropyLoss()

build_filename(opt, blacklist=['lrs', 'optim', 'gpus', 'gdot', 'vocab',
                            'f','v', 'augment', 't',
                            'save','e','l2','r', 'lr'])
logger = create_logger(opt)
pprint(opt)

optimizer = getattr(optim, opt['optim'])(model, config =
        dict(lr=opt['lr'], weight_decay=opt['l2'], momentum=opt['mom'],
            L=opt['L'], llr=lrschedule(opt, opt['e']),
            g0 = opt['g0'], g1 = opt['g1'], gdot=opt['gdot']/((ptb[0]['train'].size(0) -1) // opt['T']),
            g0max=1, g1max=10,
            beta1=opt['beta1'], clip=opt['clip'],
            verbose=opt['v'],
            t=0))

def train(e):
    optimizer.config['lr'] = lrschedule(opt, e, logger)
    #optimizer.config['llr'] = lrschedule(opt, e, logger)

    model.train()

    f, perp, dt = AverageMeter(), AverageMeter(), AverageMeter()
    fstd, perpstd = AverageMeter(), AverageMeter()
    pf, pperp = [], []

    bsz = opt['b']
    maxb = (ptb[0]['train'].size(0) -1) // opt['T']
    t0 = timer()

    n = opt['n']
    ids = deepcopy(model.ids)

    h = [model.w[i].init_hidden(opt['b']) for i in xrange(n)]
    bids = [int(random.random()*maxb) for i in xrange(n)]
    total_loss = 0

    for bi in xrange(maxb):
        _dt = timer()
        def helper():
            def feval():
                xs, ys, yhs = [None]*n, [None]*n, [None]*n
                fs = [None]*n

                for i in xrange(n):
                    # get batch and reset hidden state if dataset ends
                    x, y = batcher[i](ptb[i]['train'], bids[i]*opt['T'])
                    bids[i] += 1
                    if bids[i] > maxb:
                        bids[i] = 0
                        h[i] = model.w[i].init_hidden(opt['b'])

                    xs[i], ys[i] =  Variable(x.cuda(ids[i], async=True)), \
                            Variable(y.cuda(ids[i], async=True))
                    h[i] = models.repackage_hidden(h[i])

                for i in xrange(n):
                    yhs[i], h[i] = model.w[i](xs[i], h[i])

                for i in xrange(n):
                    fs[i] = criterion.cuda(ids[i])(yhs[i].view(-1, opt['vocab']), ys[i])

                model.backward(fs)
                th.cuda.synchronize()

                # for i in xrange(n):
                #     nn.utils.clip_grad_norm(model.w[i].parameters(), opt['clip'])
                #     for p in model.w[i].parameters():
                #         p.data.add_(-optimizer.config['lr'], p.grad.data)
                # for p1, p2 in zip(model.ref.parameters(), model.w[0].parameters()):
                #     p1.data.copy_(p2.data)

                return [fs[i].data[0] for i in xrange(n)], None, None
            return feval

        fs, _, _ = optimizer.step(helper())
        #fs, _, _ = helper()()

        f.update(np.mean(fs))
        fstd.update(np.std(fs))

        perp.update(np.mean(np.exp(fs)))
        perpstd.update(np.std(np.exp(fs)))

        pf.append(fs)
        pperp.append(np.exp(fs))

        dt.update(timer()-_dt)

        if opt['l']:
            s = dict(i=bi + e*maxb, e=e, f=np.mean(fs), perp=np.mean(np.exp(fs)),
                    fstd=np.std(fs), perpstd=np.std(np.exp(fs)), dt=dt.avg)
            logger.info('[LOG] ' + json.dumps(s))

        if bi % 25 == 0 and bi > 0:
            print((color('blue', '[%2.2fs][%2d][%4d/%4d] %2.4f+-%2.4f %2.2f+-%2.2f'))%(dt.avg,
                e,bi,maxb, np.mean(pf), np.mean(np.std(pf, 1)), np.mean(pperp), np.mean(np.std(pperp, 1))))
            pf, pperp = [], []

    if opt['l']:
        s = dict(e=e, i=0, f=f.avg, fstd=fstd.avg, perp=perp.avg, perpstd=perpstd.avg,
                train=True, t=timer()-t0)
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('blue', '++[%2d] %2.4f+-%2.4f %2.2f+-%2.2f [%2.2fs]'))% (e,
        f.avg, fstd.avg, perp.avg, perpstd.avg, timer()-t0))
    print()

        # total_loss = total_loss + fs[0]
        # bif = 200
        # if bi % bif == 0 and bi > 0:
        #     curr_loss = total_loss/float(bif)
        #     print((color('blue', '[%2.2fs][%2d][%4d/%4d] %2.4f %2.2f'))%(dt.avg,
        #         e,bi,maxb, curr_loss, math.exp(curr_loss)))
        #     total_loss = 0

def val(e, src):
    n = opt['n']

    rid = model.refid
    model.ref.eval()

    h = model.ref.init_hidden(opt['b'])
    f = 0

    for i in range(0, ptb[0][src].size(0)-1, opt['T']):
        x,y = batcher[0](ptb[0][src], i)

        x,y = Variable(x.cuda(rid), volatile=True), \
                Variable(y.cuda(rid), volatile=True)

        h = models.repackage_hidden(h)
        yh,h = model.ref(x, h)
        _f = criterion.cuda(rid)(yh.view(-1, opt['vocab']), y).data[0]
        f = f + _f*len(x)

    f = f/len(ptb[0][src])
    if opt['l']:
        s = dict(e=e, i=0, f=f, perp=math.exp(f))
        s[src] = True
        logger.info('[SUMMARY] ' + json.dumps(s))
        logger.info('')

    print((color('red', '**[%2d] %2.4f %2.4f\n'))%(e, f, math.exp(f)))
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
    opt['e'] = d['e'] + 1

    print('[Loading new optimizer]')
    optimizer = getattr(optim, opt['optim'])(model, config =
        dict(lr=opt['lr'], weight_decay=opt['l2'], momentum=opt['mom'],
            L=opt['L'], llr=lrschedule(opt, opt['e']),
            g0 = opt['g0'], g1 = opt['g1'], gdot=opt['gdot']/((ptb[0]['train'].size(0) -1) // opt['T']),
            g0max=1, g1max=10,
            verbose=opt['v'],
            t=d['t']))

    print('[Loaded model, check validation error]')
    val(d['e'], 'val')

ef = 0
try:
    for e in xrange(opt['e'], opt['B']):
        train(e)
        val(e, 'val')
        save_ensemble(e)
        ef = e
except KeyboardInterrupt:
    print('Running on test set before exiting...')

val(ef, 'test')
