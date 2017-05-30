from torch.optim import Optimizer
from copy import deepcopy
import numpy as np

import torch as th
import torch.nn as nn
from torch.nn.parallel import scatter, parallel_apply, gather

import models
import pdb

def flatten_params(model, fw, dfw):
    fw.zero_()
    dfw.zero_()
    idx = 0
    for w in model.parameters():
        n = w.numel()
        fw[idx:idx+n].copy_(w.data.view(-1))
        dfw[idx:idx+n].copy_(w.grad.data.view(-1))
        idx += n

def unflatten_params(model, fw):
    idx = 0
    for w in model.parameters():
        w.data.copy_(fw[idx:idx + w.nelement()]).view(w.size())
        idx += w.nelement()

class ESGD(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, damp=0,
                 weight_decay=0, nesterov=True,
                 L=0, eps=1e-4, g0=1e-2, g1=1e-3, rho=0,
                 sgld=False,
                 verbose=False,
                 llr=0.1, beta1=0.75)

        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(ESGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for Entropy-SGD, model and criterion'
        mf,merr = closure()

        state = self.state
        c = self.config

        if not 'N' in state:
            state['N'] = models.num_parameters(model)

        sgld = c['sgld']

        lr = c['lr']
        rho = c['rho']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = c['L']
        eps = c['eps']
        g0 = c['g0']/np.sqrt(state['N'])
        g1 = c['g1']
        verbose = c['verbose']
        llr = c['llr']
        beta1 = c['beta1']

        if not 't' in state:
            state['t'] = 0
            N = state['N']
            tmp = th.FloatTensor(N).cuda()
            state['wc'] = tmp.clone()
            state['dwc'] = tmp.clone()
            state['dw'] = tmp.clone().zero_()

            state['cache'] = {}
            cache = state['cache']
            for k in ['w', 'dw', 'mw', 'mdw']:
                state['cache'][k] = tmp.clone().zero_()

            state['eta'] = tmp.clone()
            state['mdw'] = tmp.clone().zero_()

        state['t'] += 1
        flatten_params(model, state['wc'], state['dwc'])

        g = min(g0*(1+g1)**state['t'], 10)
        cache = state['cache']
        w, dw, mw = cache['w'], cache['dw'], cache['mw']
        eta = state['eta']

        w.copy_(state['wc'])
        mw.copy_(state['wc'])

        maxf = 3.0
        cf = 0
        for i in xrange(L):
            dw.zero_()
            unflatten_params(model, w)
            cf, cerr = closure()
            flatten_params(model, w, dw)
            if wd > 0:
                dw.add_(wd, w)

            dw.add_(g, w - state['wc'])

            eta.normal_()
            dw.add_(eps/np.sqrt(0.5*llr), eta)

            if mom > 0:
                cache['mdw'].mul_(mom).add_(1-damp, dw)
                if nesterov:
                    dw.add_(mom, cache['mdw'])
                else:
                    dw = cache['mdw']

            w.add_(-llr, dw)
            mw.mul_(beta1).add_(1-beta1, w)

        dw = state['dw'].zero_()
        if L > 0:
            if rho > 0:
                dw.add_(rho, state['dwc'])
            dw.add_(state['wc'] - mw)
        else:
            dw.add_(state['dwc'])

        if sgld:
            eta.normal_()
            dw.add_(eps/np.sqrt(0.5*lr), eta)

        if verbose and state['t'] % 25 == 0:
            debug = dict(dw=dw.norm(), dwc=state['dwc'].norm(),
                dwdwc=th.dot(dw, state['dwc'])/dw.norm()/state['dwc'].norm(),
                f=cf, g=g)
            print {k : round(v, 5) for k,v in debug.items()}

        if mom > 0:
            state['mdw'].mul_(mom).add_(1-damp, dw)
            if nesterov:
                dw.add_(mom, state['mdw'])
            else:
                dw = state['mdw']

        w = state['wc']
        w.add_(-lr, dw)
        unflatten_params(model, w)

        return mf,merr

class SGD(ESGD):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, dampening=0,
             weight_decay=0, nesterov=True, L=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(SGD, self).__init__(params, config)
        self.config = config

class SGLD(ESGD):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, dampening=0,
            weight_decay=0, nesterov=True, L=0, sgld=True)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(SGLD, self).__init__(params, config)
        self.config = config

class DistESGD():
    def __init__(self, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, dampening=0,
                weight_decay=0, nesterov=True, L=0,
                g0=0.01, g1=1,
                verbose=False)

        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]
        self.config = config
        self.state = {}

    def step(self, closure=None, model=None):
        assert (closure is not None) and (model is not None), \
                'attach closure and model for DistESGD'

        state = self.state
        c = self.config

        if not 'N' in state:
            state['N'] = models.num_parameters(model.w[0])
            state['n'] = len(model.w)
            state['ids'] = deepcopy(model.ids)

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
        eps = 1e-4
        g0 = c['g0']/np.sqrt(N)
        g1 = c['g1']/np.sqrt(N)
        gdot = 1e-3
        llr = 0.1
        beta1 = 0.75

        if not 't' in state:
            state['t'] = 0
            t = th.FloatTensor(N)

            state['cache'] = {}
            state['wc'] = [t.clone().cuda(ids[i]) for i in xrange(n)]
            state['r'] = t.clone().cuda(0)

            for k in ['dw', 'dwc', 'mdw']:
                state[k] = deepcopy(state['wc'])

            cache = state['cache']
            for k in ['w', 'dw', 'mw', 'mdw']:
                cache[k] = deepcopy(state['wc'])

            for i in xrange(n):
                state['mdw'][i].zero_()
                cache['mdw'][i].zero_()

        state['t'] += 1

        cache = state['cache']
        wc, w = state['wc'], cache['w']
        dwc, dw = state['dwc'], cache['dw']
        mw, mdw = cache['mw'], cache['mdw']

        eta = state['eta']
        r = state['r'].zero_()

        # store initial w,dw
        for i in xrange(n):
            model.w[i].zero_grad()
        fs, errs = closure()
        for i in xrange(n):
            flatten_params(model.w[i], wc[i], dwc[i])

        for i in xrange(n):
            w[i].copy_(wc[i])
            mw[i].copy_(w[i])

        def feval():
            for i in xrange(n):
                model.w[i].zero_grad()
                unflatten_params(model.w[i], w[i])
            tfs, terrs = closure()
            for i in xrange(n):
                flatten_params(model.w[i], w[i], dw[i])
                if wd > 0:
                    dw[i].add_(wd, w[i])

        gsgld = min(g0*(1+gdot)**state['t'], 10)
        for l in xrange(L):
            feval()
            for i in xrange(n):
                dw[i].add_(gsgld, w[i]-wc[i])

                if mom > 0:
                    mdw[i].mul_(mom).add_(1-damp, dw[i])
                    if nesterov:
                        dw[i].add_(mom, mdw[i])
                    else:
                        dw[i] = mdw[i]

                w[i].add_(-llr, dw[i])
                mw[i].mul_(beta1).add_(1-beta1, w[i])

        r.zero_()
        r = gather(mw, 0).mul_(1/float(n))
        rc = scatter(r, ids)

        dw = state['dw']
        for i in xrange(n):
            dw[i].zero_()

        gesgd = min(g1*(1+gdot)**state['t'], 10)
        for i in xrange(n):
            if L > 0:
                dw[i].add_(wc[i]-mw[i])
            else:
                dw[i].add_(dwc[i])

            dw[i].add_(gesgd, wc[i]-rc[ids[i]])

            if mom > 0:
                state['mdw'][i].mul_(mom).add_(1-damp, dw[i])
                if nesterov:
                    dw[i].add_(mom, state['mdw'][i])
                else:
                    dw[i] = state['mdw'][i]

            wc[i].add_(-lr, dw[i])

            unflatten_params(model.ensemble[i], wc[i])

        r.zero_()
        r = gather(mw, 0).mul_(1/float(n))
        unflatten_params(model.ref, r)

        e = 1e-12
        if verbose and state['t'] % 25 == 0:
            for i in xrange(n):
                debug = dict(
                    dw=dw[i].norm(),
                    dwc=dwc[i].norm(),
                    de= 1./gesgd*(wc[i]-rc[ids[i]]).norm(),
                    dwdwc=th.dot(dw[i], dwc[i])/(dw[i].norm()+e)/(dwc[i].norm()+e),
                    wmu=th.dot(wc[i], rc[ids[i]])/(wc[i].norm()+e)/(rc[ids[i]].norm()+e),
                    gsgld=gsgld, gesgd=gesgd)
                print 'R[%2d]'%i, {k : round(v, 5) for k,v in debug.items()}

        return fs, errs