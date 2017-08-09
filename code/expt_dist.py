import torch.distributed as dist
import torch as th
import pdb, glob, sys, os, multiprocessing, argparse

opt = dict(n=3)


def run(r):
    try:
        print 'start process: ', r
        dist.init_process_group(backend='tcp',
            init_method='tcp://224.66.41.62:23456',
            world_size=opt['n'])
        print 'ran process: ', r

    except RuntimeError as e:
        sys.exit(0)
    sys.exit(0)

def spawn(r):
    os.environ['RANK'] = str(r)
    name = 'process ' + str(r)
    p = multiprocessing.Process(target=run, name=name,
                                args=(r,))
    p.start()
    return p

ps = []
for r in xrange(opt['n']):
    ps.append(spawn(r))
for p in ps:
    p.join(5)
for p in ps:
    p.terminate()
