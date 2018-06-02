import torch as th
from models import *
import torch.nn as nn
import visdom
import time

viz = visdom.Visdom()

# m = lenet({})
# mm = BBszModel({'l2':0.}, m, nn.CrossEntropyLoss())
# b = 32768
# x = th.randn(b,1,28,28)
# y = th.randint(0,10,size=(b,)).long()
# mm.forward_backward(x,y)

# yh = mm.forward(x)

# print('model')
# for p in mm.model.parameters():
#     print(p.grad.data.norm())

# print('copy')
# for p in mm.copy.parameters():
#     print(p.grad.data.norm())

# print(yh.norm())

x = np.linspace(0, np.pi, 100)
y = np.sin(x)
z = np.cos(x)

win = viz.line(np.array(0).reshape(1), name='tmp')

for i in range(100):
    viz.line(X=np.array(x[i]).reshape(1), Y=np.array(y[i]).reshape(1),
        name='train', update='append', win=win, opts=dict(showlegend=True, markers=True, colormap='Viridis'))
    viz.line(X=np.array(x[i]).reshape(1), Y=np.array(z[i]).reshape(1),
        name='val', update='append', win=win, opts=dict(showlegend=True, markers=True, colormap='Viridis'))
    time.sleep(0.1)
    print(i)