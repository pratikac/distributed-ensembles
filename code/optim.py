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

class Parle(object):
    def __init__(self, model, config = {}):

        defaults = dict(lr=0.1, mom=0.9, damp=0, llr=0.1,
                l2=0, L=25, beta1=0.0,
                g0=0.01, g1=1.0, gdot=0.5,
                g0m=1, g1m=10,
                v=False,
                t=0)
        defaults.update(**config)

        self.model = model
        self.config = deepcopy(defaults)
        self.state = dict(N=models.num_parameters(self.model.ref),
                    t=0,
                    n = len(self.model.w),
                    ids = deepcopy(self.model.ids))

    def step(self, closure=None):
        assert closure is not None, 'attach closure for Parle'

        state = self.state
        c = self.config
        model = self.model

        N = state['N']
        n = state['n']
        ids = state['ids']
        rid = model.refid

        if not 'w' in state:
            t = th.FloatTensor(N)

            state['w'] = [t.clone().cuda(ids[i]) for i in xrange(n)]
            state['dw'] = [t.clone().cuda(ids[i]) for i in xrange(n)]
            state['r'], state['dr'] = t.clone().cuda(rid), t.clone().cuda(rid)

            for i in xrange(n):
                flatten_params(model.w[i], state['w'][i], state['dw'][i])
            flatten_params(model.ref, state['r'], state['dr'])

            for k in ['mw', 'mdw', 'cmdw', 'wc', 'dwc', 'eta']:
                state[k] = [t.clone().cuda(ids[i]) for i in xrange(n)]

            for i in xrange(n):
                state['mdw'][i].zero_()
                state['cmdw'][i].zero_()

        state['t'] += 1

        w, dw = state['w'], state['dw']
        mw, mdw = state['mw'], state['mdw']
        cmdw = state['cmdw']
        eta = state['eta']

        wc, dwc = state['wc'], state['dwc']
        r, dr = state['r'], state['dr']

        def feval():
            for i in xrange(n):
                dw[i].zero_()
            cfs, cerrs, cerrs5 = closure()
            if c['l2'] > 0:
                for i in xrange(n):
                    dw[i].add_(c['l2'], w[i])
            return cfs, cerrs, cerrs5

        fs, errs, errs5 = [None]*n, [None]*n, [None]*n

        if c['L'] == 0:
            fs, errs, errs5 = feval()

        for i in xrange(n):
            wc[i].copy_(w[i])
            dwc[i].copy_(dw[i])
            mw[i].copy_(w[i])

        g = min(c['g0']*(1+c['gdot'])**state['t'], c['g0m'])
        rho = min(c['g1']*(1+c['gdot'])**state['t'], c['g1m'])
        mom = c['mom']

        for l in xrange(c['L']):
            fs, errs, errs5 = feval()
            for i in xrange(n):

                dw[i].add_(g, w[i]-wc[i])

                if c['mom'] > 0:
                    cmdw[i].mul_(mom).add_(1-c['damp'], dw[i])
                    dw[i].add_(mom, cmdw[i])

                w[i].add_(-c['llr'], dw[i])
                mw[i].mul_(c['beta1']).add_(1-c['beta1'], w[i])

        r.copy_(comm.reduce_add(mw, rid)).mul_(1/float(n))
        rc = comm.broadcast(r, ids)

        for i in xrange(n):
            if c['L'] > 0:
                dw[i].copy_(wc[i]-mw[i])
            else:
                dw[i].copy_(dwc[i])

            dw[i].add_(rho, wc[i]-rc[i])

            if c['mom'] > 0:
                mdw[i].mul_(mom).add_(1-c['damp'], dw[i])
                dw[i].add_(mom, mdw[i])

            w[i].copy_(wc[i])
            w[i].add_(-c['lr'], dw[i])

        r.copy_(comm.reduce_add(w, rid)).mul_(1/float(n))

        return fs, errs, errs5

class ElasticSGD(Parle):
    def __init__(self, model, config = {}):
        config['L'] = 0
        super(ElasticSGD, self).__init__(model, config)

class SGD(Parle):
    def __init__(self, model, config = {}):
        config['L'] = 0
        config['g1'] = 0
        super(SGD, self).__init__(model, config)

class EntropySGD(Parle):
    def __init__(self, model, config = {}):
        config['eps'] = 1e-4
        super(EntropySGD, self).__init__(model, config)

class ProxSGD(object):
    def __init__(self, model, config = {}):

        defaults = dict(lr=0.1, mom=0.9, damp=0,
                l2=0, L=25,
                g0=0.01, gdot=1e-3,
                v=False,
                t=0)
        defaults.update(**config)

        self.model = model
        self.config = deepcopy(defaults)
        self.state = dict(N=models.num_parameters(self.model.ref),
                    t=0,
                    n = len(self.model.w),
                    ids = deepcopy(self.model.ids))

        assert self.state['n'] == 1, 'prox only works for n=1'

    def step(self, closure=None):
        assert closure is not None, 'attach closure for ProxSGD'

        state = self.state
        c = self.config
        model = self.model

        N = state['N']
        n = state['n']
        ids = state['ids']
        rid = model.refid

        if not 'w' in state:
            t = th.FloatTensor(N)

            for k in ['w', 'dw', 'mdw', 'wc', 'dwc']:
                state[k] = t.clone().cuda(ids[0])
            state['r'], state['dr'], state['mdr'] = t.clone().cuda(rid), \
                        t.clone().cuda(rid), t.clone().cuda(rid)

            flatten_params(model.w[0], state['w'], state['dw'])
            flatten_params(model.ref, state['r'], state['dr'])

            state['mdw'].zero_()
            state['mdr'].zero_()

        state['t'] += 1
        g = min(c['g0']*(1+c['gdot'])**state['t'], 1)
        #mom = (state['t']-1)/(state['t']+2)
        mom = c['mom']

        w, dw, mdw = state['w'], state['dw'], state['mdw']
        wc, dwc = state['wc'], state['dwc']
        r, dr, mdr = state['r'], state['dr'], state['mdr']

        def feval():
            dw.zero_()
            cfs, cerrs, cerrs5 = closure()
            if c['l2'] > 0:
                dw.add_(c['l2'], w)
            return cfs, cerrs, cerrs5

        wc.copy_(w)
        dwc.copy_(dw)

        fs, errs, errs5 = None, None, None

        for l in xrange(c['L']):
            fs, errs, errs5 = feval()

            dw.add_(g, w-wc)

            mdw.mul_(mom).add_(1-c['damp'], dw)
            dw.add_(mom, mdw)
            w.add_(-c['lr'], dw)

        mdr.mul_(mom).add_(1-c['damp'], wc-w)
        dr.zero_()
        dr.add_(mom, mdr)
        r.copy_(wc)
        r.add_(-1, dr)

        return fs, errs, errs5

def copy_from_params(m, fw, fdw):
    # from model to fw
    fw.zero_()
    fdw.zero_()
    idx = 0
    for w in m.parameters():
        n = w.numel()
        fw[idx:idx+n].copy_(w.data.view(-1))
        if not w.grad is None:
            fdw[idx:idx+n].copy_(w.grad.data.view(-1))
        idx += n

def copy_to_params(m, fw):
    # to model from fw
    idx = 0
    for w in m.parameters():
        w.data.copy_(fw[idx:idx+w.nelement()])
        idx += w.nelement()

class FederatedParle(object):
    def __init__(self, model, config = {}):

        defaults = dict(lr=0.1, mom=0.9, damp=0, llr=0.1,
                l2=0, L=25, beta1=0.0,
                g0=0.01, g1=1.0, gdot=0.5,
                g0m=1, g1m=10,
                v=False,
                t=0)
        defaults.update(**config)

        self.model = model
        self.config = deepcopy(defaults)
        self.state = dict(N=models.num_parameters(self.model.ref),
                    t=0,
                    n = len(self.model.w),
                    ids = deepcopy(self.model.ids))

    def step(self, closure=None):
        assert closure is not None, 'attach closure for FederatedParle'

        state = self.state
        c = self.config
        model = self.model

        N = state['N']
        n = state['n']
        ids = state['ids']
        rid = model.refid

        if not 'w' in state:
            t = th.FloatTensor(N)

            state['r'] = t.clone()
            for k in ['w', 'dw', 'mdw', 'cmdw', 'wc', 'dwc']:
                state[k] = [t.clone() for i in xrange(n)]

            for i in xrange(n):
                copy_from_params(model.w[i], state['w'][i], state['dw'][i])

            for i in xrange(n):
                state['mdw'][i].zero_()
                state['cmdw'][i].zero_()

        state['t'] += 1

        w, dw = state['w'], state['dw']
        mdw = state['mdw']
        cmdw = state['cmdw']

        wc, dwc = state['wc'], state['dwc']
        r = state['r']

        def feval():
            for i in xrange(n):
                model.w[i].zero_grad()
                copy_to_params(model.w[i], w[i])

            cfs, cerrs, cerrs5 = closure()

            for i in xrange(n):
                copy_from_params(model.w[i], w[i], dw[i])

            if c['l2'] > 0:
                for i in xrange(n):
                    dw[i].add_(c['l2'], w[i])
            return cfs, cerrs, cerrs5

        fs, errs, errs5 = [None]*n, [None]*n, [None]*n

        if c['L'] == 0:
            fs, errs, errs5 = feval()

        for i in xrange(n):
            copy_from_params(model.w[i], w[i], dw[i])
            wc[i].copy_(w[i])
            dwc[i].copy_(dw[i])

        g = min(c['g0']*(1+c['gdot'])**state['t'], c['g0m'])
        rho = min(c['g1']*(1+c['gdot'])**state['t'], c['g1m'])

        for l in xrange(c['L']):
            fs, errs, errs5 = feval()
            for i in xrange(n):
                dw[i].add_(g, w[i]-wc[i])

                if c['mom'] > 0:
                    cmdw[i].mul_(c['mom']).add_(dw[i])
                    dw[i].add_(c['mom'], cmdw[i])

                w[i].add_(-c['llr'], dw[i])

        r.zero_()
        for i in xrange(n):
            r.add_(1/float(n), w[i])

        for i in xrange(n):
            if c['L'] > 0:
                dw[i].copy_(wc[i]-w[i])
            else:
                dw[i].copy_(dwc[i])

            dw[i].add_(rho, wc[i]-r)

            if c['mom'] > 0:
                mdw[i].mul_(c['mom']).add_(dw[i])
                dw[i].add_(c['mom'], mdw[i])

            w[i].copy_(wc[i])
            w[i].add_(-c['lr'], dw[i])

        r.zero_()
        for i in xrange(n):
            r.add_(1/float(n), w[i])

        for i in xrange(n):
            copy_to_params(model.w[i], w[i])
        copy_to_params(model.ref, r)

        return fs, errs, errs5