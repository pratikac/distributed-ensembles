import os, sys, subprocess, json, argparse
from itertools import product

parser = argparse.ArgumentParser(description='Quick dirty hyperoptim')
parser.add_argument('-c','--command',   help='Main command', type=str, required=True)
parser.add_argument('-p','--params',    help='JSON dict of the hyper-parameters', type=str)
parser.add_argument('-r', '--run',      help='run',  action='store_true')
parser.add_argument('-j', '--max_jobs',     help='max jobs',    type=int, default = 1)
opt = vars(parser.parse_args())

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def run_cmds(cmds, max_cmds):
    for cs in list(chunks(cmds, max_cmds)):
        ps = []
        try:
            for c in cs:
                p = subprocess.Popen(c, shell=True)
                ps.append(p)

            for p in ps:
                p.wait()

        except KeyboardInterrupt:
            print 'Killling everything'
            for p in ps:
                p.kill()
            sys.exit()

cmd = opt['command']
params = json.loads(opt['params'])

cmds = []
gs = [0,1,2]
keys,values = zip(*params.items())
for v in product(*values):
    p = dict(zip(keys,v))
    s = ''
    for k in p:
        if len(k) > 1:
            s += ' --'+k+' '+str(p[k])
        else:
            s += ' -'+k+' '+str(p[k])

    c = cmd+s
    c = c + (' -l -g %d')%(gs[len(cmds)%len(gs)])
    cmds.append(c)

if not opt['run']:
    for c in cmds:
        print c
else:
    run_cmds(cmds, opt['max_jobs'])