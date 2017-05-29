import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from torch.nn.parallel import scatter, parallel_apply, gather, replicate

opt = dict(b=128, d=0., lr=0.1, mom=0.9, n=3)

class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)

class lenet(nn.Module):
    def __init__(self, opt):
        super(lenet, self).__init__()

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.BatchNorm2d(co),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1,20,5,3,0.25),
            convbn(20,50,5,2,0.25),
            View(50*2*2),
            nn.Linear(50*2*2, 500),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(500,10),
            nn.LogSoftmax())

    def forward(self, x):
        return self.m(x)


gids = [0,1,2]

model = lenet(opt).cuda()

train_loader = th.utils.data.DataLoader(
    datasets.MNIST('/local2/pratikac/mnist', train=True, download=False,
        transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=opt['b']*opt['n'], shuffle=True, pin_memory=True)
optimizer = optim.SGD(model.parameters(), lr=opt['lr'], momentum=opt['mom'])


def train(e):
    model.train()
    for i, (x,y) in enumerate(train_loader):
        x,y = Variable(x), Variable(y.cuda(0))

        model.zero_grad()
        replicas = replicate(model, gids)
        xs = scatter([x], gids)
        yhs = parallel_apply(replicas, xs)
        yh = gather(yhs, 0)

        f = F.nll_loss(yh, y)
        f.backward()

        optimizer.step()
        if i % 10 == 0:
            print e, i, len(train_loader), round(f.data[0], 4)

for e in xrange(100):
    train(e)
    print ''