import torch as th
import models
from exptutils import *
from optim import *
from timeit import default_timer as timer
import torch.cuda.comm as comm
import torch.nn as nn
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Estimate communication latencies')
parser.add_argument('-m',
            help='lenet | allcnn | ...', type=str,
            default='lenet')
parser.add_argument('-B',
            help='num. average', type=int,
            default=100)
args = vars(parser.parse_args())

gpus = [0,1,2]
setup(t=4, s=42, gpus=gpus)

opt = dict(d=0.25, b=128, l2=0.0, dataset='cifar10', n=3, augment=False)
opt['m'] = args['m']

model = models.ReplicateModel(opt, gpus=gpus)
n = model.n
N = models.num_parameters(model.ref)
t = th.FloatTensor(N)

ids, rid = model.ids, model.refid
state = {}
dt = {}
state['w'] = [t.clone().cuda(ids[i]) for i in xrange(n)]
state['dw'] = [t.clone().cuda(ids[i]) for i in xrange(n)]
state['r'], state['dr'] = t.clone().cuda(rid), t.clone().cuda(rid)

for i in xrange(n):
    flatten_params(model.w[i], state['w'][i], state['dw'][i])
flatten_params(model.ref, state['r'], state['dr'])

t0 = timer()
x, y  = th.randn(opt['b'], 3, 32, 32), th.Tensor(opt['b']).random_(10).long()
xs, ys = [None]*n, [None]*n
B = args['B']

for i in xrange(n):
    xs[i], ys[i] =  Variable(x.cuda(ids[i], async=True)), \
                    Variable(y.squeeze().cuda(ids[i], async=True))
for b in xrange(B):
    fs, errs, errs5 = model.forward(xs, ys)
    model.backward(fs)
dt['compute'] = (timer() - t0)/float(B)
print '[compute]: ', dt['compute']

t0 = timer()
B = args['B']
for i in xrange(B):
    state['r'].copy_(comm.reduce_add(state['w'], rid)).mul_(1/float(n))
dt['comm'] = (timer() - t0)/float(B)
print '[comm]: ', dt['comm']

print 'ratio: ', dt['comm']/dt['compute']