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
            help='mnistfc | lenet | allcnn', type=str,
            default='mnistfc')
parser.add_argument('-l',
            help='location', type=str,
            default='/Users/pratik/Dropbox/siap17data')
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
dc = loaddir(os.path.join(opt['l'], opt['m']), force=opt['f'])
dc = dc[(dc['summary'] == True)]
dc = dc.filter(items=['optim', 'f', 'L', 'e', 'top1', 's','train','val'])
dc.loc[dc.L==0,'L'] = 1
dc.loc[:,'e'] += 1
dc['ee'] = dc['e']*dc['L']

dc = dc.filter(items=['optim', 'f', 'ee', 'top1', 's','train','val'])

# remove the diverging last few epochs for PME
# dc = dc.drop(dc[(dc.optim == 'PME') & (dc.ee > 150)].index)

dc = dc[(dc.optim != 'PME') & (dc.optim != 'LL')]

colors = {'SGD':'k', 'ESGD':'r', 'H-ESGD':'b', 'HJ':'g', 'PME':'y', \
        'HEAT':'m','LL':'c'}

def rough(dc, train=False):
    d = dc.copy()
    d = d[(d['val'] == True)]
    plt.figure(1)
    plt.clf()
    sns.tsplot(time='ee',value='top1',data=d,
                unit='s',condition='optim', color=colors)
    sns.tsplot(time='ee',value='top1',
                data=d[ (d['optim'] != 'SGD')],
                marker='o', interpolate=False,
                unit='s',condition='optim', color=colors,
                legend=False)
    plt.title(opt['m'])
    plt.grid('on')

    if train:
        plt.figure(2)
        plt.clf()
        d = dc.copy()
        d = d[(d['train'] == True)]
        d = d.drop_duplicates(['s', 'ee'])
        sns.tsplot(time='ee',value='top1',data=d,
                    unit='s',condition='optim', color=colors,
                    legend=False, linestyle='--')
        sns.tsplot(time='ee',value='top1',
                    data=d[ (d['optim'] != 'SGD')],
                    marker='o',
                    unit='s',condition='optim', color=colors,
                    legend=False, linestyle='--')
        plt.title(opt['m'])
        plt.grid('on')

def mnistfc():
    fig = plt.figure(1, figsize=(8,7))
    plt.clf()
    ax = fig.add_subplot(111)

    rough(dc[(dc['ee']<=200)])
    plt.legend(loc='best')
    plt.xlabel(r'Epochs $\times$ L')
    plt.ylabel(r'\% Error')
    plt.ylim([1.0, 1.5])
    plt.xlim([20, 200])
    xt = [20, 50, 100, 150, 200]
    plt.xticks(xt, [str(s) for s in xt])
    yt = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    plt.yticks(yt, [str(s) for s in yt])
    # plt.title(r'mnistfc: Validation error')
    plt.title('')

    plt.plot(range(200), 1.08*np.ones(200), 'k--', lw=1)
    plt.text(50, 1.09, r'$1.08$\%', fontsize=fsz,
            verticalalignment='center', color='k')
    if opt['s']:
        plt.savefig('../fig/mnistfc_valid.pdf', bbox_inches='tight')

def lenet():
    fig = plt.figure(1, figsize=(8,7))
    plt.clf()
    ax = fig.add_subplot(111)

    rough(dc[(dc['ee']<100)])
    plt.legend(loc='best')
    plt.xlabel(r'Epochs $\times$ L')
    plt.ylabel(r'\% Error')
    plt.ylim([0.5, 1.0])
    plt.xlim([0, 80])
    xt = [0, 20,40,60,80]
    plt.xticks(xt, [str(s) for s in xt])
    yt = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.yticks(yt, [str(s) for s in yt])
    # plt.title(r'LeNet: Validation error')
    plt.title('')
    # plt.plot(range(80), 0.5*np.ones(80), 'k--', lw=1)
    # ax.text(25, 0.49, r'$0.5$\%', fontsize=fsz,
    #         verticalalignment='center', color='k')

    if opt['s']:
        plt.savefig('../fig/lenet_valid.pdf', bbox_inches='tight')


def allcnn():
    fig = plt.figure(1, figsize=(8,7))
    plt.clf()
    fig.add_subplot(111)

    rough(dc[(dc['ee']<200)])
    plt.legend(loc='best')
    plt.xlabel(r'Epochs $\times$ L')
    plt.ylabel(r'\% Error')
    plt.ylim([5, 20])
    plt.xlim([0, 200])
    xt = [0, 50, 100, 150, 200]
    plt.xticks(xt, [str(s) for s in xt])
    yt = [5,10,15,20]
    plt.yticks(yt, [str(s) for s in yt])
    # plt.title(r'All-CNN: Validation error')
    plt.title('')

    plt.plot(range(200), 7.82*np.ones(200), 'k--', lw=1)
    plt.text(20, 8.25, r'$7.82$\%', fontsize=fsz,
            verticalalignment='center', color='k')

    if opt['s']:
        plt.savefig('../fig/allcnn_valid.pdf', bbox_inches='tight')

    fig = plt.figure(2, figsize=(8,7))
    plt.clf()
    ax = fig.add_subplot(111)

    for o in sorted(colors.keys()):
        d = dc.copy()
        if o == 'LL' or o == 'PME':
            continue

        d2 = d[(d['train'] == True) & (d['optim'] == o)]
        d2 = d2.drop_duplicates(['s', 'ee'])
        sns.tsplot(time='ee',value='f',data=d2,
                    unit='s',condition='optim', color=colors[o])
        if o != 'SGD':
            sns.tsplot(time='ee',value='f',
                        data=d2,
                        marker='o',
                        unit='s',condition='optim', color=colors[o],
                        legend=False)
    plt.grid('on')

    plt.legend(loc='best')
    plt.xlabel(r'Epochs $\times$ L')
    plt.ylabel(r'$f(x)$')
    plt.ylim([0, 0.6])
    plt.xlim([0, 200])
    xt = [0, 50, 100, 150, 200]
    plt.xticks(xt, [str(s) for s in xt])
    yt = [0, 0.2, 0.4, 0.6]
    plt.yticks(yt, [str(s) for s in yt])
    plt.title('')
    # plt.title(r'All-CNN: Training loss')

    plt.plot(range(200), 0.046*np.ones(200), 'k--', lw=1)

    ax.text(20, 0.06, r'$0.046$', fontsize=fsz,
            verticalalignment='center', color='k')

    if opt['s']:
        plt.savefig('../fig/allcnn_loss.pdf', bbox_inches='tight')

if opt['r']:
    rough(dc, train=True)
    sys.exit(0)
else:
    globals()[opt['m']]()
    sys.exit(0)
