import torch as th
import sys
import subprocess

args = list(sys.argv[1:])

opt=dict()
assert(args[0] == '-n')
opt['n'] = int(args[1])
args = args[2:]

workers = []
for i in range(opt['n']):
    if '-r' in args:
        args[args.index('-r')+1] = str(i)
    else:
        args.append('-r')
        args.append(str(i))

    if '-n' in args:
        args[args.index('-n')+1] = str(opt['n'])
    else:
        args.append('-n')
        args.append(str(opt['n']))

    p = subprocess.Popen([str(sys.executable)] + args)
    workers.append(p)

try:
    for p in workers:
        p.wait()

except KeyboardInterrupt:
    print('Killing all subprocess')
    for p in workers:
        p.kill()
    sys.exit()