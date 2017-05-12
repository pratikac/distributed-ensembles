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
        mf,merr = closure()

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

        state['t'] += 1
        flatten_params(model, state['wc'], state['dwc'])

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

        # binarize
        thresh = nn.Threshold(1,0)
        w = thresh(w).sign_() - thresh(-w).sign_()

        unflatten_params(model, w)
        mf,merr = closure()

        return mf,merr

class ElasticSGD(Optimizer):
    # params here is a dummy, we will handle everything manually
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.1, momentum=0.9, dampening=0,
                weight_decay=0, nesterov=True,
                g0=1e-2, g1=0,
                verbose=False,
                llr=0.1)

        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(ElasticSGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None):
        assert (closure is not None) and (model is not None), \
                'attach closure for ElasticSGD, replicated model'

        state = self.state
        c = self.config

        if not 'N' in state:
            state['N'] = models.num_parameters(model.ensemble[0])
            state['n'] = len(model.ensemble)

        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['dampening']
        nesterov = c['nesterov']
        verbose = c['verbose']
        g0 = c['g0']
        g1 = c['g1']

        if not 't' in state:
            state['t'] = 0
            N = state['N']
            tmp = th.FloatTensor(N)
            state['wc'] = [tmp.clone().cuda(model.gidxs[i]) for i in xrange(state['n'])]
            state['dwc'] = [tmp.clone().cuda(model.gidxs[i]) for i in xrange(state['n'])]
            state['mdw'] = [tmp.clone().cuda(model.gidxs[i]).zero_() for i in xrange(state['n'])]

            state['mu'] = tmp.clone().cuda(0)

            cache = state['cache']
            # copies of mu on each GPU
            cache['mu'] = tmp.clone().cuda(0)
            cache['rmu'] = [tmp.clone().cuda(model.gidxs[i]) for i in xrange(state['n'])]

        state['t'] += 1
        g = g0*(1+g1)**state['t']

        for i in xrange(state['n']):
            model.ensemble[i].zero_grad()
        fs, errs = closure()

        w = state['wc']
        dw, mdw = state['dwc'], state['mdw']
        mu = state['mu'].zero_()
        cmu, crmu = state['cache']['mu'], state['cache']['rmu']

        for i in xrange(state['n']):
            flatten_params(model.ensemble[i], w[i], dw[i])
            if wd > 0:
                dw[i].add_(wd, w[i])

            cmu.copy_(w[i])
            mu.add_(1/float(state['n']), cmu)

        # hack, when we want to use output coupling
        # if g0 < 1e-12:
        #     mu.copy_(w[0])

        for i in xrange(state['n']):
            crmu[i].copy_(mu)

        if verbose and state['t'] % 25 == 0:
            debug = dict()
            debug['mu'] = mu.norm()
            for i in xrange(state['n']):
                debug['dw'+str(i)] = dw[i].norm()
                debug['de'+str(i)] = (w[i]-crmu[i]).norm()
                debug['ol'+str(i)] = th.dot(w[i], crmu[i])/w[i].norm()/(crmu[i].norm() + 1e-6)
                debug['ddwmu'+str(i)] = th.dot(dw[i],w[i]-crmu[i])/(dw[i].norm()+1e-6)/((w[i]-crmu[i]).norm() + 1e-6)
            debug['g'] = g
            print {k : round(v, 5) for k,v in debug.items()}

        unflatten_params(model.reference, mu)
        for i in xrange(state['n']):
            dw[i].add_(g, w[i]-crmu[i])

            if mom > 0:
                mdw[i].mul_(mom).add_(1-damp, dw[i])
                if nesterov:
                    dw[i].add_(mom, mdw[i])
                else:
                    dw[i] = mdw[i]

            w[i].add_(-lr, dw[i])

            unflatten_params(model.ensemble[i], w[i])

        return fs, errs

