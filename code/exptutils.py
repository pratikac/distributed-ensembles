import os, pdb, sys, json, subprocess
import numpy as np
import time, logging, pprint

import torch as th
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
    blacklist = blacklist + ['l','h','o','b','B','g','retrain']
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
        print 'Retraining, will stop logging'
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res