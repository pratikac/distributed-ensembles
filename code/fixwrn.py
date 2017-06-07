import json
import numpy as np

# d = []
# for l in open('/Users/pratik/Dropbox/cs269data/wrn-demo.txt'):
#     d.append(json.loads(l[12:-1]))


# r = []
# for i in xrange(len(d)):
#     di = d[i]
#     r.append(dict(e=di['epoch']-1, top1=100.-di['train_acc'], f=di['train_loss'], i=0, train=True))
#     r.append(dict(e=di['epoch']-1, top1=100.-di['test_acc'], f=di['test_loss'], i=0, val=True))


# with open('wrnfix.log', 'w') as f:
#     for ri in r:
#         f.write('[SUMMARY] ' + json.dumps(ri)+'\n')


d = []
for l in open('/Users/pratik/Dropbox/cs269data/wrn/sgd/szagoruyko_wrnlog.txt'):
    if l[:4] == 'json':
        d.append(json.loads(l[12:-1]))
    else:
        continue

r = []
for i in xrange(len(d)):
    di = d[i]
    #r.append(dict(e=di['epoch']-1, top1=100.-di['train_acc'], f=di['loss'], i=0, train=True))
    r.append(dict(e=di['epoch']-1, top1=100.-di['test_acc'], f=0, i=0, val=True))
with open('szagoruyko.log', 'w') as f:
    for ri in r:
        f.write('[SUMMARY] ' + json.dumps(ri)+'\n')
