import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, logging, pdb
from copy import deepcopy
import exptutils
import numpy as np
from torch.nn.parallel import scatter, parallel_apply, gather

class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

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

class allcnn(nn.Module):
    def __init__(self, opt = {'d':0.5}, c1=96, c2=192):
        super(allcnn, self).__init__()
        self.name = 'allcnn'

        if opt['d'] < 0:
            opt['d'] = 0.5
        if opt['l2'] < 0:
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

class caddtable_t(nn.Module):
    def __init__(self, m1, m2):
        super(caddtable_t, self).__init__()
        self.m1, self.m2 = m1, m2

    def forward(self, x):
        return th.add(self.m1(x), self.m2(x))

class wideresnet(nn.Module):
    @staticmethod
    def block(ci, co, s, p=0.):
        h = nn.Sequential(
                nn.BatchNorm2d(ci),
                nn.ReLU(inplace=True),
                nn.Conv2d(ci, co, kernel_size=3, stride=s, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Dropout(p),
                nn.Conv2d(co, co, kernel_size=3, stride=1, padding=1, bias=False))
        if ci == co:
            return caddtable_t(h, nn.Sequential())
        else:
            return caddtable_t(h,
                        nn.Conv2d(ci, co, kernel_size=1, stride=s, padding=0, bias=False))

    @staticmethod
    def netblock(nl, ci, co, blk, s, p=0.):
        ls = [blk(i==0 and ci or co, co, i==0 and s or 1, p) for i in xrange(nl)]
        return nn.Sequential(*ls)

    def __init__(self, opt = {'d':0., 'depth':28, 'widen':10}):
        super(wideresnet, self).__init__()
        self.name = 'wideresnet'

        opt['d'] = 0.
        opt['l2'] = 5e-4
        d, depth, widen = opt['d'], opt['depth'], opt['widen']

        if opt['dataset'] == 'cifar10':
            num_classes = 10
        elif opt['dataset'] == 'cifar100':
            num_classes = 100

        nc = [16, 16*widen, 32*widen, 64*widen]
        assert (depth-4)%6 == 0, 'Incorrect depth'
        n = (depth-4)/6

        self.m = nn.Sequential(
                nn.Conv2d(3, nc[0], kernel_size=3, stride=1, padding=1, bias=False),
                self.netblock(n, nc[0], nc[1], self.block, 1, d),
                self.netblock(n, nc[1], nc[2], self.block, 2, d),
                self.netblock(n, nc[2], nc[3], self.block, 2, d),
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
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                #m.weight.data.normal_(0, math.sqrt(2./m.in_features))
                m.bias.data.zero_()

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class ReplicateModel(nn.Module):
    def __init__(self, opt, gpus):
        super(ReplicateModel, self).__init__()

        self.t = 0
        self.n = opt['n']
        n = self.n

        self.ids = [gpus[i%len(gpus)] for i in xrange(n)]
        self.w = [globals()[opt['m']](opt).cuda(self.ids[i]) for i in xrange(n)]
        self.ref = globals()[opt['m']](opt).cuda(0)

    def forward(self, xs, ys):
        xs = [[a] for a in xs]
        return parallel_apply(self.w, xs)

    def backward(self, fs):
        f = sum(gather(fs, 0))
        f.backward()

    def train(self):
        self.ref.train()
        for i in xrange(self.n):
            self.w[i].train()

    def eval(self):
        self.ref.eval()
        for i in xrange(self.n):
            self.w[i].eval()