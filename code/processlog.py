import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

import os, sys, glob2, pdb, re, json
import cPickle as pickle
sns.set()

colors = sns.color_palette("husl", 8)

whitelist = set(['s','m','lr','eps', 'g0', 'g1', 'L', 'optim'])

def get_params_from_filename(s):
    t = s[s.find('('):s.find('_opt_')]
    _s = s[s.find('_opt_')+5:-4]
    r = json.loads(_s)
    r = {k: v for k,v in r.items() if k in whitelist}
    r['t'] = t
    return r

def loadlog(f):
    logs, summary = [], []
    opt = get_params_from_filename(f)

    for l in open(f):
        if '[LOG]' in l[:5]:
            logs.append(json.loads(l[5:-1]))
        elif '[SUMMARY]' in l[:9]:
            summary.append(json.loads(l[9:-1]))
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

def loaddir(dir, expr='*', force=False):
    pkl = dir+'/log.p'

    if (not force) and os.path.isfile(pkl):
        return pickle.load(open(pkl, 'r'))

    fs = sorted(glob2.glob(dir + '/*/' + expr + '.log'))
    d = []

    for f in fs:
        di = loadlog(f)
        d.append(di)
        print get_params_from_filename(f)

    d = pd.concat(d)
    pickle.dump(d, open(pkl, 'w'), protocol=pickle.HIGHEST_PROTOCOL)
    return d