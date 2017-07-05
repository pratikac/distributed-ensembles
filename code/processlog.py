import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

import os, sys, glob2, pdb, re, json
import cPickle as pickle
sns.set()

colors = sns.color_palette("husl", 8)

blacklist = ['filename', 'o', 'save', 'B', 'lrs', 'validate', 'g','f','widen',
            'v', 'l2', 'lr','l','augment','depth','retrain','e','optim']

def get_params_from_log(f):
    r = {}
    for l in open(f):
        if '[OPT]' in l[:5]:
            r = json.loads(l[5:-1])
            fn = r['filename']
            for k in blacklist:
                r.pop(k, None)
            #r = {k: v for k,v in r.items() if k in whitelist}
            r['t'] = fn[fn.find('(')+1:fn.find(')')]
            return r
    assert len(r.keys) > 0, 'Could not find [OPT] marker in '+f

def loadlog(f):
    logs, summary = [], []
    opt = get_params_from_log(f)

    for l in open(f):
        if '[LOG]' in l[:5]:
            logs.append(json.loads(l[5:-1]))
        elif '[SUMMARY]' in l[:9]:
            try:
                summary.append(json.loads(l[9:-1]))
            except ValueError, e:
                pdb.set_trace()
        else:
            try:
                s = json.loads(l)
            except:
                continue
            if s['i'] == 0:
                if not 'val' in s:
                    s['train'] = True
                summary.append(s)
            else:
                logs.append(s)
    dl, ds = pd.DataFrame(logs), pd.DataFrame(summary)

    dl['log'] = True
    ds['summary'] = True
    for k in opt:
        dl[k] = opt[k]
        ds[k] = opt[k]
    d = pd.concat([dl, ds])
    return d

def loaddir(dir, expr='/*/*', force=False):
    pkl = dir+'/log.p'

    if (not force) and os.path.isfile(pkl):
        return pickle.load(open(pkl, 'r'))

    fs = sorted(glob2.glob(dir + expr + '.log'))
    d = []

    for f in fs:
        di = loadlog(f)
        d.append(di)
        print get_params_from_log(f)

    d = pd.concat(d)
    pickle.dump(d, open(pkl, 'w'), protocol=pickle.HIGHEST_PROTOCOL)
    return d
