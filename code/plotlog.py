import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob, pdb, argparse
import cPickle as pickle
import seaborn as sns

from processlog import *

sns.set_style('ticks')
sns.set_color_codes()

parser = argparse.ArgumentParser(description='Plotter')
parser.add_argument('-m',
            help='lenet | allcnn', type=str,
            default='lenet')
parser.add_argument('-l',
            help='location', type=str,
            default='/Users/pratik/Dropbox/cs269data')
parser.add_argument('-f',
            help='reprocess data',
            action='store_true')
parser.add_argument('-s',
            help='save figures',
            action='store_true')
parser.add_argument('-r',
            help='rough plots',
            action='store_true')
opt = vars(parser.parse_args())

if opt['s']:
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

if not opt['r']:
    fsz = 24
    plt.rc('font', size=fsz)
    plt.rc('axes', titlesize=fsz)
    plt.rc('axes', labelsize=fsz)
    plt.rc('xtick', labelsize=fsz)
    plt.rc('ytick', labelsize=fsz)
    plt.rc('legend', fontsize=fsz*0.8)
    plt.rc('figure', titlesize=fsz)

# load data
whitelist = ['n', 'L', 'e',
            'f', 'fstd', 'top1', 'top1std',
            's',
            'train','val',
            'frac']

dc = loaddir(os.path.join(opt['l'], opt['m']), force=opt['f'])
dc = dc[(dc['summary'] == True)]
dc = dc.filter(items=whitelist)

dc['frac'].fillna(1.0, inplace=True)
dc['n'].fillna(1, inplace=True)
dc['top1std'].fillna(0.0, inplace=True)
dc['fstd'].fillna(0.0, inplace=True)

dc.loc[dc.L==0,'L'] = 1
dc.loc[:,'e'] += 1
dc['ee'] = dc['e']*dc['L']

d = dc.copy()
d = d[(d['val'] == True)]

colors = {'SGD':'k', 'ESGD':'r', 'Dist-ESGD (n=3)':'b', 'Dist-ESGD (n=6)':'g'}

d['optim'] = 'SGD'
d.ix[(d['L'] !=1) & (d['n'] == 1), 'optim'] = 'ESGD'
for ni in np.unique(dc.n):
    if ni != 1:
        d.ix[(d['L'] !=1) & (d['n'] == ni), 'optim'] = 'Dist-ESGD (n=%d)'%(ni)

d = d.filter(items=['f','top1','s','ee','optim','n'])

def rough(d):
    fig = plt.figure()
    plt.clf()
    sns.tsplot(time='ee',value='top1',data=d,
                unit='s',condition='optim', color=colors)
    sns.tsplot(time='ee',value='top1',
                data=d[(d['optim'] != 'SGD')],
                marker='o', interpolate=False,
                unit='s',condition='optim', color=colors,
                legend=False)
    plt.title(opt['m'])
    plt.grid('on')

    plt.xlabel('Epochs x L')
    plt.legend(markerscale=0)
    return fig

# mnist
f = rough(d)
plt.figure(f.number)
plt.title('LeNet (full data)')
plt.xlim([0, 100])
plt.ylim([0.4, 1.0])
if opt['s']:
    plt.savefig('../fig/lenet_valid.pdf', bbox_inches='tight')