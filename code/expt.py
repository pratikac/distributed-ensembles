from models import *
import torch as th
import torch.nn as nn
from operator import itemgetter
from future.utils import iteritems
import numpy as np
import collections
import pdb

m = lenet({})
#m = nn.Sequential(nn.Linear(5,3,2))

def flatten_params(model, flattened):
    flattened.zero_()
    idx = 0
    for w in model.parameters():
        n = w.numel()
        flattened[idx:idx+n].copy_(w.data.view(-1))
        idx += n

def unflatten_params(model, flattened):
    offset = 0
    for param in model.parameters():
        param.data.copy_(flattened[offset:offset + param.nelement()]).view(param.size())
        offset += param.nelement()

def check_flatten():
    for i in xrange(1000):
        x = flatten_params(m)
        unflatten_params(m, x)
    return x

N = num_parameters(m)
x = th.FloatTensor(N).zero_()
flatten_params(m, x)

class caddtable_t(nn.Module):
    def __init__(self, ms):
        super(caddtable_t, self).__init__()
        self.ms = ms
    def forward(self, x):
        o = self.ms[0](x)
        for i in xrange(1, len(self.ms)):
            o += self.ms[i](x)
        return o
