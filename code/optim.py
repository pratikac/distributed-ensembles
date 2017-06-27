from copy import deepcopy
import numpy as np

import torch as th
import torch.nn as nn
import torch.cuda.comm as comm
from torch.autograd import Variable

import models
import pdb

def flatten_params(m, fw, fdw):
    idx = 0

    for w in m.parameters():
        n = w.numel()
        fw[idx:idx+n].copy_(w.data.view(-1))
        w.data.set_(fw.storage(), idx, w.size())
        if w.grad is None:
            w._grad = Variable(w.data.clone())
            w._grad.data.set_(fdw.storage(), idx, w.size())
        else:
            fdw[idx:idx+n].copy_(w.grad.data.view(-1))
            w.grad.data.set_(fdw.storage(), idx, w.size())

        idx += w.data.numel()

class DistESGD(object):
    def __init__(self, model, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, dampening=0, llr=0.1,
                weight_decay=0, nesterov=True, L=25, beta1=0.75,
                g0=0.01, g1=1, gdot=0.5, eps=0,
                verbose=False,
                t=0)

        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        self.model = model
        self.config = config
        self.state = dict(N=models.num_parameters(self.model.ref),
                    t=0,
                    n = len(self.model.w),
                    ids = deepcopy(self.model.ids))

    def step(self, closure=None):
        assert closure is not None, 'attach closure for DistESGD'

        state = self.state
        c = self.config
        model = self.model

        N = state['N']
        n = state['n']
        ids = state['ids']
        rid = model.refid

        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['dampening']
        nesterov = c['nesterov']
        verbose = c['verbose']
        L = c['L']
        g0 = c['g0']
        g1 = c['g1']
        gdot = c['gdot']
        llr = c['llr']
        beta1 = c['beta1']
        eps = c['eps']

        if not 'w' in state:
            t = th.FloatTensor(N)

            state['w'] = [t.clone().cuda(ids[i]) for i in xrange(n)]
            state['dw'] = [t.clone().cuda(ids[i]) for i in xrange(n)]
            state['r'], state['dr'], state['mdr'], state['cr'] = t.clone().cuda(rid), \
                    t.clone().cuda(rid), t.clone().cuda(rid), t.clone().cuda(rid)

            for i in xrange(n):
                flatten_params(model.w[i], state['w'][i], state['dw'][i])
            flatten_params(model.ref, state['r'], state['dr'])

            for k in ['mw', 'mdw', 'cmdw', 'wc', 'dwc', 'eta']:
                state[k] = [t.clone().cuda(ids[i]) for i in xrange(n)]

            for i in xrange(n):
                state['mdw'][i].zero_()
                state['cmdw'][i].zero_()
            state['mdr'].zero_()

        state['t'] += 1

        w, dw = state['w'], state['dw']
        mw, mdw = state['mw'], state['mdw']
        cmdw = state['cmdw']
        eta = state['eta']

        wc, dwc = state['wc'], state['dwc']
        r, dr, mdr = state['r'], state['dr'], state['mdr']

        def feval():
            for i in xrange(n):
                dw[i].zero_()
            cfs, cerrs, cerrs5 = closure()
            if wd > 0:
                for i in xrange(n):
                    dw[i].add_(wd, w[i])
            return cfs, cerrs, cerrs5

        fs, errs, errs5 = [None]*n, [None]*n, [None]*n
        if L == 0:
            fs, errs, errs5 = feval()

        for i in xrange(n):
            wc[i].copy_(w[i])
            dwc[i].copy_(dw[i])
            mw[i].copy_(w[i])

        gsgld = min(g0*(1+gdot)**state['t'], 1)
        gesgd = min(g1*(1+gdot)**state['t'], 10)

        for l in xrange(L):
            fs, errs, errs5 = feval()
            for i in xrange(n):

                dw[i].add_(gsgld, w[i]-wc[i])

                if eps > 0:
                    eta[i].normal_()
                    dw[i].add_(eps, eta[i])

                if mom > 0:
                    cmdw[i].mul_(mom).add_(1-damp, dw[i])
                    if nesterov:
                        dw[i].add_(mom, cmdw[i])
                    else:
                        dw[i] = cmdw[i]

                w[i].add_(-llr, dw[i])
                mw[i].mul_(beta1).add_(1-beta1, w[i])

        # update reference with mw
        state['cr'].copy_(r)
        r.copy_(comm.reduce_add(mw, rid)).mul_(1/float(n))
        rc = comm.broadcast(r, ids)

        for i in xrange(n):
            if L > 0:
                dw[i].copy_(wc[i]-mw[i])
            else:
                dw[i].copy_(dwc[i])

            dw[i].add_(gesgd, wc[i]-rc[i])

            if mom > 0:
                mdw[i].mul_(mom).add_(1-damp, dw[i])
                if nesterov:
                    dw[i].add_(mom, mdw[i])
                else:
                    dw[i] = mdw[i]

            w[i].copy_(wc[i])
            w[i].add_(-lr, dw[i])

        if False:
            dr.zero_()
            dr.copy_(comm.reduce_add(w, rid)).mul_(-1/float(n))
            dr.add_(1, state['cr'])
            if mom > 0:
                mdr.mul_(mom).add_(1-damp, dr)
                if nesterov:
                    dr.add_(mom, mdr)
                else:
                    dr.copy_(mdr)
            r.copy_(state['cr'])
            r.add_(-lr, dr)
        else:
            r.copy_(comm.reduce_add(w, rid)).mul_(1/float(n))

        e = 1e-12
        if verbose and state['t'] % 25 == 0:
            for i in xrange(n):
                debug = dict(
                    dw=dw[i].norm(),
                    dwc=dwc[i].norm(),
                    de= 1./gesgd*(w[i]-rc[i]).norm(),
                    dwdwc=th.dot(dw[i], dwc[i])/(dw[i].norm()+e)/(dwc[i].norm()+e),
                    wmu=th.dot(w[i], rc[i])/(w[i].norm()+e)/(rc[i].norm()+e),
                    gsgld=gsgld, gesgd=gesgd)
                print 'R[%2d]'%i, {k : round(v, 5) for k,v in debug.items()}

        return fs, errs, errs5

class ElasticSGD(DistESGD):
    def __init__(self, model, config = {}):
        config['L'] = 0
        super(ElasticSGD, self).__init__(model, config)

class SGD(DistESGD):
    def __init__(self, model, config = {}):
        config['L'] = 0
        config['g1'] = 0
        super(SGD, self).__init__(model, config)

class EntropySGD(DistESGD):
    def __init__(self, model, config = {}):
        config['eps'] = 1e-4
        super(EntropySGD, self).__init__(model, config)