import torch as th
from models import *
import torch.nn as nn

m = lenet({})
mm = BBszModel({'l2':0.}, m, nn.CrossEntropyLoss())
b = 32768
x = th.randn(b,1,28,28)
y = th.randint(0,10,size=(b,)).long()
mm.forward_backward(x,y)

yh = mm.forward(x)

print('model')
for p in mm.model.parameters():
    print(p.grad.data.norm())

print('copy')
for p in mm.copy.parameters():
    print(p.grad.data.norm())

# print(yh.norm())