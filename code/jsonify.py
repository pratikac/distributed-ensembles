import torch as th
import json, sys
from exptutils import *
import numpy as np
from collections import OrderedDict
import glob

opt = add_args([
['--json', '', 'json file, convert to pz'],
['--pz', '', 'pz file, convert to json']
])

if opt['pz'] != '':
    d = th.load(opt['pz'])['state_dict']
    for k in d:
        d[k] = d[k].cpu().numpy().tolist()
    json.dump(d, open(opt['pz']+'.json', 'wb'))

if opt['json'] != '':
    d = OrderedDict(json.load(open(opt['json'], 'rb')))
    for k in d:
        d[k] = th.from_numpy(np.array(d[k], dtype=np.float32))
    th.save(dict(state_dict=d, name='allcnn'), opt['json']+'.pz')

# for f in glob.glob('/local2/pratikac/results/allcnn_ensemble/allcnn_s_*'):
#     print f
#     d = th.load(f)['state_dict']
#     for k in d:
#         d[k] = d[k].cpu().numpy().tolist()
#     json.dump(d, open(f+'.json', 'wb'))