class admmSGD(object):
    def __init__(self, model, config = {}):

        defaults = dict(lr=0.1, lrd=0, momentum=0.9, dampening=0,
                weight_decay=0, nesterov=True, L=25,
                g0=0.01, g1=1, gdot=1e-3, eps=0, num_batches=500,
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
        verbose = c['verbose']
        L = c['L']
        g0 = c['g0']
        gdot = c['gdot']/c['num_batches']
        llr = 0.1



        state['t'] += 1
