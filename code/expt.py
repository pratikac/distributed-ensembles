import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from models import *
from optim import *

def flatten_params(m, fw, fdw):
    fw.zero_()
    fdw.zero_()
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

m = lenet({})
w = th.FloatTensor(num_parameters(m))
dw = th.FloatTensor(num_parameters(m))
flatten_params(m, w, dw)

print 'Setting param storage'
print w[:25].view(5,5)
print list(m.parameters())[0][0]

x = Variable(th.randn(1,1,28,28))
y = Variable(th.LongTensor(1).random_(10))
yh = m(x)
f = nn.CrossEntropyLoss()(yh, y)
f.backward()
print yh, y, f

print 'Checking grad storage'
print dw[:25].view(5,5)
print list(m.parameters())[0].grad[0]
