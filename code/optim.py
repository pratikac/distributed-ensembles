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

class ESGD(object):
    def __init__(self, model, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, damp=0,
                 weight_decay=0, nesterov=True,
                 L=0, eps=1e-4, g0=1e-2, g1=1e-3,
                 verbose=False,
                 llr=0.1, beta1=0.75)

        self.model = model

        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        self.config = config
        self.state = dict(N=models.num_parameters(self.model), t=0)

    def step(self, closure=None):
        assert closure is not None, 'attach closure for Entropy-SGD'

        state = self.state
        c = self.config

        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = c['L']
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']
        verbose = c['verbose']
        llr = c['llr']
        beta1 = c['beta1']

        if not 'w' in state:
            N = state['N']
            tmp = th.FloatTensor(N).cuda()

            for k in ['w', 'dw', 'mw', 'mdw', 'cmdw', 'eta', 'wc', 'dwc']:
                state[k] = tmp.clone()

            flatten_params(self.model, state['w'], state['dw'])
            state['mdw'].zero_()
            state['cmdw'].zero_()

        state['t'] += 1

        g = min(g0*(1+g1)**state['t'], 1)

        w, dw = state['w'], state['dw']
        mw, cmdw, eta = state['mw'], state['cmdw'], state['eta']
        mdw = state['mdw']

        f, err = None, None
        if L == 0:
            f, err = closure()

        wc, dwc = state['wc'], state['dwc']
        wc.copy_(w)
        dwc.copy_(dw)
        mw.copy_(w)

        for i in xrange(L):
            dw.zero_()
            f, err = closure()
            if wd > 0:
                dw.add_(wd, w)

            dw.add_(g, w - wc)

            eta.normal_()
            dw.add_(eps/np.sqrt(0.5*llr), eta)

            if mom > 0:
                cmdw.mul_(mom).add_(1-damp, dw)
                if nesterov:
                    dw.add_(mom, cmdw)
                else:
                    dw = cmdw

            w.add_(-llr, dw)
            mw.mul_(beta1).add_(1-beta1, w)

        if L > 0:
            dw.copy_(wc - mw)
        else:
            dw.copy_(dwc)

        if verbose and state['t'] % 25 == 0:
            debug = dict(dw=dw.norm(), dwc=dwc.norm(),
                dwdwc=th.dot(dw, dwc)/dw.norm()/dwc.norm(),
                f=cf, g=g)
            print {k : round(v, 5) for k,v in debug.items()}

        if mom > 0:
            mdw.mul_(mom).add_(1-damp, dw)
            if nesterov:
                dw.add_(mom, mdw)
            else:
                dw = mdw

        w.copy_(wc)
        w.add_(-lr, dw)

        return f, err

class SGD(ESGD):
    def __init__(self, model, config = {}):

        defaults = dict(L=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(SGD, self).__init__(model, config)

class HJ(ESGD):
    def __init__(self, model, config = {}):

        defaults = dict(beta1=0, eps=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(HJ, self).__init__(model, config)

class DistESGD(object):
    def __init__(self, model, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, dampening=0,
                weight_decay=0, nesterov=True, L=25,
                g0=0.01, g1=1,
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

        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['dampening']
        nesterov = c['nesterov']
        verbose = c['verbose']
        L = c['L']
        g0 = c['g0']
        g1 = c['g1']
        gdot = 1e-3
        llr = 0.1
        beta1 = 0.75

        if not 'w' in state:
            t = th.FloatTensor(N)

            state['w'] = [t.clone().cuda(ids[i]) for i in xrange(n)]
            state['dw'] = [t.clone().cuda(ids[i]) for i in xrange(n)]
            state['r'], state['dr'] = t.clone().cuda(0), t.clone().cuda(0)

            for i in xrange(n):
                flatten_params(model.w[i], state['w'][i], state['dw'][i])
            flatten_params(model.ref, state['r'], state['dr'])

            for k in ['mw', 'mdw', 'cmdw', 'wc', 'dwc']:
                state[k] = [t.clone().cuda(ids[i]) for i in xrange(n)]

            for i in xrange(n):
                state['mdw'][i].zero_()
                state['cmdw'][i].zero_()

        state['t'] += 1

        w, dw = state['w'], state['dw']
        mw, mdw = state['mw'], state['mdw']
        cmdw = state['cmdw']

        wc, dwc = state['wc'], state['dwc']
        r = state['r']

        def feval():
            for i in xrange(n):
                dw[i].zero_()
            cfs, cerrs = closure()
            if wd > 0:
                for i in xrange(n):
                    dw[i].add_(wd, w[i])
            return cfs, cerrs

        fs, errs = [None]*n, [None]*n
        if L == 0:
            fs, errs = feval()

        for i in xrange(n):
            wc[i].copy_(w[i])
            dwc[i].copy_(dw[i])
            mw[i].copy_(w[i])

        gsgld = min(g0*(1+gdot)**state['t'], 1)
        for l in xrange(L):
            fs, errs = feval()
            for i in xrange(n):
                dw[i].add_(gsgld, w[i]-wc[i])

                if mom > 0:
                    cmdw[i].mul_(mom).add_(1-damp, dw[i])
                    if nesterov:
                        dw[i].add_(mom, cmdw[i])
                    else:
                        dw[i] = cmdw[i]

                w[i].add_(-llr, dw[i])
                mw[i].mul_(beta1).add_(1-beta1, w[i])

        r.zero_()
        r.copy_(comm.reduce_add(mw, 0)).mul_(1/float(n))
        rc = comm.broadcast(r, ids)

        gesgd = min(g1*(1+gdot)**state['t'], 1)
        for i in xrange(n):
            if L > 0:
                dw[i].copy_(wc[i]-mw[i])
            else:
                dw[i].copy_(dwc[i])

            dw[i].add_(gesgd, wc[i]-rc[ids[i]])

            if mom > 0:
                mdw[i].mul_(mom).add_(1-damp, dw[i])
                if nesterov:
                    dw[i].add_(mom, mdw[i])
                else:
                    dw[i] = mdw[i]

            w[i].copy_(wc[i])
            w[i].add_(-lr, dw[i])

        r.zero_()
        r.copy_(comm.reduce_add(w, 0)).mul_(1/float(n))

        e = 1e-12
        if verbose and state['t'] % 25 == 0:
            for i in xrange(n):
                debug = dict(
                    dw=dw[i].norm(),
                    dwc=dwc[i].norm(),
                    de= 1./gesgd*(w[i]-rc[ids[i]]).norm(),
                    dwdwc=th.dot(dw[i], dwc[i])/(dw[i].norm()+e)/(dwc[i].norm()+e),
                    wmu=th.dot(w[i], rc[ids[i]])/(w[i].norm()+e)/(rc[ids[i]].norm()+e),
                    gsgld=gsgld, gesgd=gesgd)
                print 'R[%2d]'%i, {k : round(v, 5) for k,v in debug.items()}

        return fs, errs

class ElasticSGD(DistESGD):
    def __init__(self, model, config = {}):

        defaults = dict(L=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(ElasticSGD, self).__init__(model, config)