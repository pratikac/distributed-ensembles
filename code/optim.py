from torch.optim import Optimizer
from copy import deepcopy
import numpy as np

import torch as th
import torch.nn as nn

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
                 L=0, eps=1e-4, g0=1e-2, g1=0, rho=0,
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
        g0 = c['g0']
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

        g = g0*(1+g1)**state['t']

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

class BSGD(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, dampening=0,
             weight_decay=0, nesterov=True, L=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(BSGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for BSGD, model and criterion'

        state = self.state
        c = self.config

        if not 'N' in state:
            state['N'] = models.num_parameters(model)

        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['dampening']
        nesterov = c['nesterov']
        verbose = c['verbose']

        if not 't' in state:
            state['t'] = 0
            N = state['N']
            tmp = th.FloatTensor(N).cuda()
            state['wc'] = tmp.clone()
            state['dwc'] = tmp.clone()
            state['mdw'] = tmp.clone().zero_()
            state['bw'] = tmp.clone().cpu()   # binary w
            state['bwc'] = tmp.clone()

        state['t'] += 1
        state['bw'].copy_(state['wc'])

        # binary connect
        # tmp = th.from_numpy(np.random.binomial(n=1, p=np.clip((state['bw'].numpy()+1)/2., 0, 1),\
        #             size=(state['N'],))).float()
        # state['bw'].fill_(-1).add_(2, tmp)
        state['bw'].sign_().mul_(0.1)
        unflatten_params(model, state['bw'])
        mf, merr = closure()
        flatten_params(model, state['bwc'], state['dwc'])

        dw = state['dwc']
        if wd > 0:
            dw.add_(wd, state['wc'])

        if mom > 0:
            state['mdw'].mul_(mom).add_(1-damp, dw)
            if nesterov:
                dw.add_(mom, state['mdw'])
            else:
                dw = state['mdw']

        w = state['wc']
        w.add_(-lr, dw)

        if verbose and state['t'] % 100 == 0:
            debug = dict(dw=dw.norm(),
                w=w.norm(),
                wmu=w.mean(),
                wstd=w.std(),
                gmu=dw.mean(),
                gstd=dw.std(),
                ell2=state['wc'].norm()*wd,
                f=mf)
            print {k : round(v, 5) for k,v in debug.items()}

        #state['bw'].copy_(w).sign_().mul_(0.1)
        unflatten_params(model, state['wc'])

        return mf,merr

class DistributedESGD():
    def __init__(self, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, dampening=0,
                weight_decay=0, nesterov=True, L=0,
                g00=1e-2, g01=0,
                g10=1e-2, g11=0,
                verbose=False,
                llr=0.1, eps=1e-4)

        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]
        self.config = config
        self.state = {}

    def step(self, closure=None, model=None):
        assert (closure is not None) and (model is not None), \
                'attach closure for DistributedESGD, replicated model'

        state = self.state
        c = self.config

        if not 'N' in state:
            state['N'] = models.num_parameters(model.ensemble[0])
            state['n'] = len(model.ensemble)

        n = state['n']

        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['dampening']
        nesterov = c['nesterov']
        verbose = c['verbose']
        L = c['L']
        eps = c['eps']
        g00 = c['g00']
        g01 = c['g01']
        g10 = c['g10']
        g11 = c['g11']
        verbose = c['verbose']
        llr = c['llr']
        beta1 = 0.75

        if not 't' in state:
            state['t'] = 0
            N = state['N']
            tmp = th.FloatTensor(N)
            state['wc'] = [tmp.clone().cuda(model.gidxs[i]) for i in xrange(state['n'])]
            state['dw'] = [tmp.clone().cuda(model.gidxs[i]) for i in xrange(state['n'])]
            state['dwc'] = [tmp.clone().cuda(model.gidxs[i]) for i in xrange(state['n'])]
            state['mdw'] = [tmp.clone().cuda(model.gidxs[i]).zero_() for i in xrange(state['n'])]

            state['mu'] = tmp.clone()
            state['rmu'] = [tmp.clone().cuda(model.gidxs[i]) for i in xrange(state['n'])]
            state['eta'] = [tmp.clone().cuda(model.gidxs[i]) for i in xrange(state['n'])]

            state['cache'] = {}
            cache = state['cache']
            for k in ['w', 'dw', 'mw', 'mdw']:
                cache[k] = [tmp.clone().cuda(model.gidxs[i]) for i in xrange(state['n'])]
            for i in xrange(n):
                cache['mdw'][i].zero_()

        state['t'] += 1

        cache = state['cache']
        wc, w = state['wc'], cache['w']
        dwc, dw = state['dwc'], cache['dw']
        mw, mdw = cache['mw'], cache['mdw']

        eta = state['eta']
        rmu = state['rmu']
        mu = state['mu'].zero_()
        for i in xrange(n):
            rmu[i].zero_()

        # store initial w,dw
        for i in xrange(n):
            model.ensemble[i].zero_grad()
        fs, errs = closure()
        for i in xrange(n):
            flatten_params(model.ensemble[i], wc[i], dwc[i])

        for i in xrange(n):
            w[i].copy_(wc[i])
            mw[i].copy_(w[i])

        def get_all_gradients():
            for i in xrange(n):
                model.ensemble[i].zero_grad()
                unflatten_params(model.ensemble[i], w[i])
            tfs, terrs = closure()
            for i in xrange(n):
                flatten_params(model.ensemble[i], w[i], dw[i])
                if wd > 0:
                    dw[i].add_(wd, w[i])

        g0 = g00*(1+g01)**state['t']
        for l in xrange(L):
            get_all_gradients()

            for i in xrange(n):
                dw[i].add_(g0, w[i]-wc[i])

                eta[i].normal_()
                dw[i].add_(eps/np.sqrt(0.5*llr), eta[i])

                if mom > 0:
                    mdw[i].mul_(mom).add_(1-damp, dw[i])
                    if nesterov:
                        dw[i].add_(mom, mdw[i])
                    else:
                        dw[i] = mdw[i]

                w[i].add_(-llr, dw[i])
                mw[i].mul_(beta1).add_(1-beta1, w[i])

        mu.zero_()
        for i in xrange(n):
            mu.add_(1/float(n), mw[i].cpu())
        for i in xrange(n):
            rmu[i].copy_(mu)

        dw = state['dw']
        for i in xrange(n):
            dw[i].zero_()

        g1 = g10*(1+g11)**state['t']
        for i in xrange(n):
            if L > 0:
                dw[i].add_(wc[i]-mw[i])
            else:
                dw[i].add_(dwc[i])

            dw[i].add_(g1, wc[i]-rmu[i])

            if mom > 0:
                state['mdw'][i].mul_(mom).add_(1-damp, dw[i])
                if nesterov:
                    dw[i].add_(mom, state['mdw'][i])
                else:
                    dw[i] = state['mdw'][i]

            wc[i].add_(-lr, dw[i])

            unflatten_params(model.ensemble[i], wc[i])

        mu.zero_()
        for i in xrange(n):
            mu.add_(1/float(n), wc[i].cpu())
        unflatten_params(model.reference, mu)

        if verbose and state['t'] % 25 == 0:
            for i in xrange(n):
                debug = dict(dw=dw[i].norm(), dwc=dwc[i].norm(), de= g0*(wc[i]-rmu[i]).norm(),
                    dwdwc=th.dot(dw[i], dwc[i])/(dw[i].norm()+1e-6)/(dwc[i].norm()+1e-6),
                    wmu=th.dot(wc[i], rmu[i])/(wc[i].norm()+1e-6)/(rmu[i].norm()+1e-6),
                    g0=g0, g1=g1)
                print 'R[%2d]'%i, {k : round(v, 5) for k,v in debug.items()}

        return fs, errs