from __future__ import print_function
import os, pdb, sys, json, subprocess
import numpy as np
import time, logging, pprint

import torch as th
import torchnet as tnt
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse

colors = {  'red':['\033[1;31m','\033[0m'],
            'blue':['\033[1;34m','\033[0m']}

def color(c, s):
    return colors[c][0] + s + colors[c][1]

def add_args(args):
    p = argparse.ArgumentParser('')
    # [key, default, help, {action_store etc.}]
    for a in args:
        if len(a) == 2:
            a += ['', {}]
        elif len(a) == 3:
            a.append({})
        a[3]['help'] = a[2]

        if type(a[1]) == bool:
            if a[1]:
                a[3]['action'] = 'store_false'
            else:
                a[3]['action'] = 'store_true'
        else:
            a[3]['type'] = type(a[1])
            a[3]['default'] = a[1]

        p.add_argument(a[0], **a[3])
    return vars(p.parse_args())

def build_filename(opt, blacklist=[], marker=''):
    blacklist = blacklist + ['l','h','o','B','g','r']
    o = json.loads(json.dumps(opt))
    for k in blacklist:
        o.pop(k,None)

    t = ''
    if not marker == '':
        t = marker + '_'
    t = t + time.strftime('(%b_%d_%H_%M_%S)') + '_opt_'
    opt['filename'] = t + json.dumps(o, sort_keys=True,
                separators=(',', ':'))

def opt_from_filename(s, ext='.log'):
    _s = s[s.find('_opt_')+5:-len(ext)]
    d = json.loads(_s)
    d['time'] = s[s.find('('):s.find(')')][1:-1]
    return d

def gitrev(opt):
    cmds = [['git', 'rev-parse', 'HEAD'],
            ['git', 'status'],
            ['git', 'diff']]
    rs = []
    for c in cmds:
        subp = subprocess.Popen(c,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
        r, _ = subp.communicate()
        rs.append(r)

    rs[0] = rs[0].strip()
    return rs

def create_logger(opt, idx=0):
    if not opt['l']:
        return

    if len(opt.get('retrain', '')) > 0:
        print('Retraining, will stop logging')
        return

    if opt.get('filename', None) is None:
        build_filename(opt)

    d = opt.get('o','/local2/pratikac/results')
    fn = os.path.join(d, opt['filename']+'.log')
    l = logging.getLogger('%s'%idx)
    l.propagate = False

    fh = logging.FileHandler(fn)
    fmt = logging.Formatter('%(message)s')
    fh.setFormatter(fmt)
    l.setLevel(logging.INFO)
    l.addHandler(fh)

    r = gitrev(opt)
    l.info('SHA %s'%r[0])
    l.info('STATUS %s'%r[1])
    l.info('DIFF %s'%r[2])

    l.info('')
    l.info('[OPT] ' + json.dumps(opt))
    l.info('')

    return l

def save(model, opt, marker=''):
    d = opt.get('o','/local2/pratikac/results')
    #fn = os.path.join(d, opt['filename']+'.pz')
    fn = os.path.join(d, model.name+'_'+marker+'.pz')

    o = {   'state_dict': model.state_dict(),
            'name': model.name}
    th.save(o, fn)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

def clerr(output, target, topk=(1,)):
    r = [100.0 - a for a in accuracy(output, target, topk)]
    if len(r) == 1:
        return r[0]
    return r

def setup(t=1, s=42):
    np.random.seed(s)
    th.manual_seed(s)
    if th.cuda.device_count() > 0:
        th.cuda.manual_seed_all(s)

def dry_feed(m, loader, mid=0, opt=None):
    def set_dropout(cache = None, p=0):
        if cache is None:
            cache = []
            for l in m.modules():
                if 'Dropout' in str(type(l)):
                    cache.append(l.p)
                    l.p = p
            return cache
        else:
            for l in m.modules():
                if 'Dropout' in str(type(l)):
                    assert len(cache) > 0, 'cache is empty'
                    l.p = cache.pop(0)

    m.train()
    cache = set_dropout()
    for bi, (x,y) in enumerate(loader):
        x =   Variable(x.cuda(mid), volatile=True)
        yh = m(x)
    set_dropout(cache)
    m.eval()

def schedule(opt, e, logger=None, k='lr'):
    ks = k + 's'
    if opt[ks] == '':
        opt[ks] = json.dumps([[opt['B'], opt[k]]])

    rs = json.loads(opt[ks])

    idx = len(rs)-1
    for i in range(len(rs)):
        if e < rs[i][0]:
            idx = i
            break
    r = rs[idx][1]

    print('[%s]: '%k, r)
    if opt['l'] and logger:
        logger.info('[%s] '%k + json.dumps({'%s'%k: r}))
    return r

def lrschedule(opt, e, logger=None):
    return schedule(opt, e, logger, 'lr')

def Lschedule(opt, e, logger=None):
    return schedule(opt, e, logger, 'L')

class AverageMeters(object):
    def __init__(self, ks):
        self.m = {}
        for k in ks:
            self.m[k] = tnt.meter.AverageValueMeter()
    def add(self, v):
        for k in v:
            assert k in self.m, 'Key not found'
            self.m[k].add(v[k])
    def value(self):
        return {k:self.m[k].value()[0] for k in self.m}
    def reset(self):
        for k in ks:
            self.m[k].reset()

from line_profiler import LineProfiler
def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    return inner