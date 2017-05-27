import torch as th
import torch.nn as nn
from torch.autograd import Variable
import math, logging, pdb
from copy import deepcopy
import exptutils
import numpy as np

class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)

def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])

class mnistfc(nn.Module):
    def __init__(self, opt):
        super(mnistfc, self).__init__()
        self.name = 'mnsitfc'

        c = 1024
        opt['d'] = 0.2
        opt['l2'] = 0.

        self.m = nn.Sequential(
            View(784),
            nn.Dropout(0.2),
            nn.Linear(784,c),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(c),
            nn.Dropout(opt['d']),
            nn.Linear(c,c),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(c),
            nn.Dropout(opt['d']),
            nn.Linear(c,10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)
        logging.info(s)


    def forward(self, x):
        return self.m(x)

class small_mnistfc(nn.Module):
    def __init__(self, opt):
        super(small_mnistfc, self).__init__()
        self.name = 'small_mnsitfc'

        c = 400
        opt['d'] = 0.0
        opt['l2'] = opt['l2']

        self.m = nn.Sequential(
            View(784),
            nn.Dropout(0.2),
            nn.Linear(784,c),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(c),
            nn.Dropout(opt['d']),
            nn.Linear(c,c),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(c),
            nn.Dropout(opt['d']),
            # nn.Linear(c,c),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(c),
            # nn.Dropout(opt['d']),
            nn.Linear(c,10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)


class lenet(nn.Module):
    def __init__(self, opt):
        super(lenet, self).__init__()
        self.name = 'lenet'
        opt['d'] = 0.25
        opt['l2'] = 0.

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.BatchNorm2d(co),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1,20,5,3,opt['d']),
            convbn(20,50,5,2,opt['d']),
            View(50*2*2),
            nn.Linear(50*2*2, 500),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(500,10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class rotlenet(nn.Module):
    def __init__(self, opt):
        super(rotlenet, self).__init__()
        self.name = 'rotlenet'
        opt['d'] = 0.3

        def convpool(ci,co,ksz,psz,pstr,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=pstr),
                nn.Dropout(p))
        def conv(ci,co,ksz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.Dropout(p))

        self.m = nn.Sequential(
            conv(1,20,3,opt['d']),
            convpool(20,20,3,2,2,0),
            conv(20,20,3,opt['d']),
            conv(20,20,3,opt['d']),
            conv(20,20,3,opt['d']),
            conv(20,20,3,opt['d']),
            conv(20,10,4,0),
            View(10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)


class allcnn(nn.Module):
    def __init__(self, opt = {'d':0.5}, c1=96, c2=192):
        super(allcnn, self).__init__()
        self.name = 'allcnn'

        if opt['d'] < 1e-6:
            opt['d'] = 0.5
        if opt['l2'] < 1e-6:
            opt['l2'] = 1e-3

        if opt['dataset'] == 'cifar10':
            num_classes = 10
        elif opt['dataset'] == 'cifar100':
            num_classes = 100

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.BatchNorm2d(co),
                nn.ReLU(True))
        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,num_classes,1,1),
            nn.AvgPool2d(8),
            View(num_classes))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class small_allcnn(allcnn):
    def __init__(self, opt = {'d':0.5}, c1=32, c2=64):
        self.name = 'small_allcnn'

        opt['d'] = 0.25
        opt['l2'] = 1e-3

        super(small_allcnn, self).__init__(opt, c1, c2)

class caddtable_t(nn.Module):
    def __init__(self, m1, m2):
        super(caddtable_t, self).__init__()
        self.m1, self.m2 = m1, m2

    def forward(self, x):
        return self.m1(x) + self.m2(x)

class wideresnet(nn.Module):
    def __init__(self, opt = {'d':0., 'depth':28, 'widen':10}):
        super(wideresnet, self).__init__()
        self.name = 'wideresnet'

        opt['d'] = 0.
        opt['depth'] = 28
        opt['widen'] = 10
        opt['l2'] = 5e-4
        d, depth, widen = opt['d'], opt['depth'], opt['widen']

        if opt['dataset'] == 'cifar10':
            num_classes = 10
        elif opt['dataset'] == 'cifar100':
            num_classes = 100

        nc = [16, 16*widen, 32*widen, 64*widen]
        assert (depth-4)%6 == 0, 'Incorrect depth'
        n = (depth-4)/6

        def block(ci, co, s, p=0.):
            h = nn.Sequential(
                    nn.Sequential(nn.BatchNorm2d(ci),
                    nn.ReLU(inplace=True)),
                    nn.Conv2d(ci, co, kernel_size=3, stride=s, padding=1, bias=False),
                    nn.BatchNorm2d(co),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(co, co, kernel_size=3, stride=1, padding=1, bias=False))
            if ci == co:
                return caddtable_t(h, nn.Sequential())
            else:
                return caddtable_t(h,
                        nn.Conv2d(ci, co, kernel_size=1, stride=s, padding=0, bias=False))

        def netblock(nl, ci, co, blk, s, p=0.):
            ls = [blk(i==0 and ci or co, co, i==0 and s or 1, p) for i in xrange(nl)]
            return nn.Sequential(*ls)

        self.m = nn.Sequential(
                nn.Conv2d(3, nc[0], kernel_size=3, stride=1, padding=1, bias=False),
                netblock(n, nc[0], nc[1], block, 1, d),
                netblock(n, nc[1], nc[2], block, 2, d),
                netblock(n, nc[2], nc[3], block, 2, d),
                nn.BatchNorm2d(nc[3]),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(8),
                View(nc[3]),
                nn.Linear(nc[3], num_classes))

        for m in self.m.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2./m.in_features))
                m.bias.data.zero_()

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class RNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, param):
        super(RNN, self).__init__()
        xdim, hdim, nlayers = param['vocab'], param['hdim'], \
                param.get('layers',2)
        self.encoder = nn.Embedding(xdim, hdim)
        self.rnn = getattr(nn, param['m'])(hdim, hdim, nlayers,
                    dropout=param['d'])
        self.decoder = nn.Linear(hdim, xdim)
        self.drop = nn.Dropout(param['d'])

        if param['tie']:
            self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.rnn_type = param['m']
        self.hdim = hdim
        self.nlayers = nlayers

    def init_weights(self):
        dw = 0.1
        self.encoder.weight.data.uniform_(-dw, dw)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-dw, dw)

    def forward(self, x, h):
        f = self.drop(self.encoder(x))
        yh, hh = self.rnn(f, h)
        yh = self.drop(yh)
        decoded = self.decoder(yh.view(yh.size(0)*yh.size(1), yh.size(2)))
        return decoded.view(yh.size(0), yh.size(1), decoded.size(1)), hh

    def init_hidden(self, bsz):
        w = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(w.new(self.nlayers, bsz, self.hdim).zero_()),
                    Variable(w.new(self.nlayers, bsz, self.hdim).zero_()))
        else:
            return Variable(w.new(self.nlayers, bsz, self.hdim).zero_())

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class ptbs(RNN):
    def __init__(self, opt={}):
        self.name = 'ptbs'
        hdim = opt.get('hdim', 200)
        d = opt.get('d', 0.2)
        param = dict(vocab=opt['vocab'], hdim=hdim, layers=2,
                d=d, tie=True, m='LSTM')

        super(ptbs, self).__init__(param)

class ptbl(RNN):
    def __init__(self, opt={}):
        self.name = 'ptbl'
        hdim = opt.get('hdim', 1500)
        d = opt.get('d', 0.65)

        param = dict(vocab=opt['vocab'], hdim=hdim, layers=2,
                d=d, tie=True, m='LSTM')

        super(ptbl, self).__init__(param)


class ReplicateModel(nn.Module):
    def __init__(self, opt, crit, crit_coupling, gidxs):
        super(ReplicateModel, self).__init__()

        self.t = 0
        self.opt = deepcopy(opt)
        self.n = opt['n']
        n = self.n
        self.reference = globals()[opt['m']](opt)

        self.ensemble = [globals()[opt['m']](opt) for i in xrange(n)]
        self.criteria = [deepcopy(crit) for i in xrange(n)]
        self.criteria_coupling = [deepcopy(crit_coupling) for i in xrange(n)]
        self.gidxs = [gidxs[i%len(gidxs)] for i in xrange(n)]

        self.fs = [None for i in xrange(n)]
        self.fklds = [None for i in xrange(n)]
        self.ftots = [None for i in xrange(n)]

        self.errs = [None for i in xrange(n)]

        self.reference.cuda(self.gidxs[0])
        for i in xrange(self.n):
            self.ensemble[i].cuda(self.gidxs[i])
            self.criteria[i].cuda(self.gidxs[i])
            self.criteria_coupling[i].cuda(self.gidxs[i])

    def forward(self, xs, ys):
        yhs = [None for i in xrange(self.n)]
        for i in xrange(self.n):
            yhs[i] = self.ensemble[i](xs[i])

        for i in xrange(self.n):
            f = self.criteria[i](yhs[i], ys[i])
            prec1, = exptutils.accuracy(yhs[i].data, ys[i].data, topk=(1,))

            self.fs[i] = f
            self.errs[i] = 100.-prec1[0]

        for i in xrange(self.n):
            self.ftots[i] = self.fs[i]

        if self.opt['alpha'] > 0:
            yhs_softmax = sum([nn.LogSoftmax()(yhs[i].cpu()*self.opt['beta']) \
                        for i in xrange(self.n)])/float(self.n)
            yhs_softmax = yhs_softmax.data
            ensemble_avg = yhs_softmax.clone().zero_()
            ensemble_avg.addcdiv_(1, yhs_softmax, yhs_softmax.sum(1).expand_as(yhs_softmax))

            for i in xrange(self.n):
                self.fklds[i] = self.criteria_coupling[i](
                        nn.LogSoftmax()(yhs[i]), \
                        Variable(ensemble_avg.clone().cuda(self.gidxs[i]))
                        )
                self.ftots[i] += self.opt['alpha']*self.fklds[i]/float(self.n)

            freq = 25 if self.opt['L'] == 0 else 25*self.opt['L']
            if self.opt['v'] and self.t % freq == 0:
                #print 'softmax: ', [round(ensemble_avg.mean(0).squeeze()[k], 3) for k in xrange(10)]
                print 'KLD: ', [round(kld.data[0],5) for kld in self.fklds]

        self.t += 1
        return self.fs, self.errs

    def backward(self):
        for i in xrange(self.n):
            self.ftots[i].backward()

    def train(self):
        self.reference.train()
        for i in xrange(self.n):
            self.ensemble[i].train()

    def eval(self):
        self.reference.eval()
        for i in xrange(self.n):
            self.ensemble[i].eval()
