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

# if opt['s']:
#     from matplotlib import rc
#     rc('font',**{'family':'serif','serif':['Palatino']})
#     rc('text', usetex=True)

if not opt['r']:
    fsz = 18
    plt.rc('font', size=fsz)
    plt.rc('axes', titlesize=fsz)
    plt.rc('axes', labelsize=fsz)
    plt.rc('xtick', labelsize=fsz)
    plt.rc('ytick', labelsize=fsz)
    plt.rc('legend', fontsize=fsz)
    plt.rc('figure', titlesize=fsz)

def set_ticks(xt=[], xts=[], yt=[], yts=[]):
    if len(xt):
        if not len(xts):
            xts = [str(s) for s in xt]
        plt.xticks(xt, xts)
    if len(yt):
        if not len(yts):
            yts = [str(s) for s in yt]
        plt.yticks(yt, yts)

# load data
whitelist = ['n', 'L', 'e',
            'f', 'fstd', 'top1', 'top1std',
            's',
            'train','val',
            'frac']
colors = {  'SGD':'k', 'SGD (full)':'k',
            'Entropy-SGD':'r',
            'Elastic-SGD (n=3)':'b', 'Elastic-SGD (n=6)':'b', 'Elastic-SGD (n=8)':'b',
            'Parle (n=3)':'g', 'Parle (n=6)':'g', 'Parle (n=8)':'m'}

normalize_params = dict(
    wrn_cifar10 = {'nb': 391, 'SGD': 0.28, 'Elastic-SGD (n=3)': 0.6, 'Entropy-SGD': 6.8, 'Parle': 12.5},
    wrn_cifar100 = {'nb': 391, 'SGD': 0.28, 'Elastic-SGD (n=3)': 0.6, 'Entropy-SGD': 6.8, 'Parle': 12.5},
    wrn_svhn = {'nb': 4722, 'SGD': 0.04, 'Elastic-SGD (n=3)': 0.08, 'Entropy-SGD': 1.02, 'Parle': 1.89},
    allcnn_cifar10 = {'nb': 391, 'SGD': 0.03, 'Elastic-SGD (n=3)': 0.06, 'Elastic-SGD (n=6)': 0.06, 'Entropy-SGD': 0.67, 'Parle': 1.29},
    allcnn_cifar100 = {'nb': 391, 'SGD': 0.03, 'Elastic-SGD (n=3)': 0.06, 'Elastic-SGD (n=6)': 0.06, 'Entropy-SGD': 0.67, 'Parle': 1.29},
    lenet = {'nb': 469, 'SGD': 0.009, 'Elastic-SGD (n=3)': 0.007, 'Elastic-SGD (n=6)': 0.015, 'Entropy-SGD': 0.21, 'Parle': 0.18},
)

# fix Parle
for n in [3,6,8]:
    for k in normalize_params:
        normalize_params[k]['Parle (n=%d)'%n] = normalize_params[k]['Parle']

df = loaddir(os.path.join(opt['l'], opt['m']), force=opt['f'])
df = df[(df['summary'] == True)]
df = df.filter(items=whitelist)

df['frac'].fillna(1.0, inplace=True)
df['n'].fillna(1, inplace=True)
df['top1std'].fillna(0.0, inplace=True)
df['fstd'].fillna(0.0, inplace=True)
#df['top5std'].fillna(0.0, inplace=True)

df.loc[df.L==0,'L'] = 1
df.loc[:,'e'] += 1

df['optim'] = 'GD'
df.loc[(df['L'] == 1) & (df['n'] == 1), 'optim'] = 'SGD'
df.loc[(df['L'] != 1) & (df['n'] == 1), 'optim'] = 'Entropy-SGD'
for ni in np.unique(df.n):
    if ni != 1:
        df.loc[(df['L'] != 1) & (df['n'] == ni), 'optim'] = 'Parle (n=%d)'%(ni)
        df.loc[(df['L'] == 1) & (df['n'] == ni), 'optim'] = 'Elastic-SGD (n=%d)'%(ni)

df['t'] = df['e']*df['frac']
for optim in df.optim.unique():
    per_epoch = normalize_params[opt['m']]['nb']*normalize_params[opt['m']][optim]
    df.loc[df['optim'] == optim, 't'] *= per_epoch
df['t'] = df['t']/60.0

sgd = df[(df['optim']=='SGD') & (df['frac'] == 1.0)].copy()
sgd.replace({'SGD':'SGD (full)'}, inplace=True)

def rough(d, idx=1):
    dc = d.copy()
    dc = dc.filter(items=['f','top1','s','t','optim','n','val','train'])

    # val
    fig = plt.figure(idx, figsize=(8,7))
    plt.clf()

    dv = dc[(dc['val'] == True)]
    sns.tsplot(time='t',value='top1',data=dv,
                unit='s',condition='optim', color=colors)
    sns.tsplot(time='t',value='top1',
                data=   dv[(dv['optim'] != 'SGD')&
                        (dv['optim'] != 'SGD (full)')&
                        (dv['optim'] != 'Elastic-SGD (n=3)')&
                        (dv['optim'] != 'Elastic-SGD (n=6)')&
                        (dv['optim'] != 'Elastic-SGD (n=8)')],
                marker='o', interpolate=False,
                unit='s',condition='optim', color=colors,
                legend=False)

    # # train
    # dt = dc[(dc['train'] == True)]
    # sns.tsplot(time='t',value='top1',data=dt,
    #             unit='s',condition='optim', color=colors,
    #             legend=False)
    # sns.tsplot(time='t',value='top1',
    #             data=dt[(dt['optim'] != 'SGD')],
    #             marker='o', interpolate=False,
    #             unit='s',condition='optim', color=colors,
    #             legend=False)

    plt.grid('on')
    plt.legend(markerscale=0)
    return fig

def lenet():
    f = rough(df[df['frac'] == 1.0], 1)
    plt.figure(f.number)
    plt.title(r'LeNet: MNIST')
    plt.xlabel(r'wall-clock time (min)')
    plt.ylabel(r'top1 error ($\%$)')
    plt.xlim([0, 8])
    plt.ylim([0.4, 1.0])
    set_ticks(xt=[0,2,4,6,8], yt=[0.4, 0.6, 0.8, 1.0])

    plt.text(2, 0.425, r'$0.44$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=6)'])
    plt.text(4, 0.62, r'$0.48$', fontsize=fsz,
        verticalalignment='center', color=colors['Elastic-SGD (n=6)'])
    plt.text(5, 0.57, r'$0.49$', fontsize=fsz,
        verticalalignment='center', color=colors['Entropy-SGD'])
    plt.text(6.5, 0.57, r'$0.50$', fontsize=fsz,
        verticalalignment='center', color=colors['SGD'])
    if opt['s']:
        plt.savefig('../fig/lenet_full_valid.pdf', bbox_inches='tight')

def allcnn_cifar10():
    f = rough(df[df['frac'] == 1], 1)
    plt.figure(f.number)
    plt.title(r'All-CNN: CIFAR-10 (full)')
    plt.xlabel(r'wall-clock time (min)')
    plt.ylabel(r'top1 error ($\%$)')
    plt.xlim([0, 75])
    plt.ylim([5, 14])
    set_ticks(xt=[0, 25, 50, 75], yt=[5,8,11,14])

    plt.text(65, 7, r'$5.18$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=3)'])
    plt.text(40, 5.18, r'$5.76$', fontsize=fsz,
        verticalalignment='center', color=colors['Entropy-SGD'])
    plt.text(35, 7, r'$6.15$', fontsize=fsz,
        verticalalignment='center', color=colors['SGD'])

    if opt['s']:
        plt.savefig('../fig/allcnn_cifar10_full_valid.pdf', bbox_inches='tight')

    sgd['frac'].replace(1.0, 0.5, inplace=True)
    df2 = df[ (df['frac'] == 0.5)
            & (df['optim'] != 'Entropy-SGD')
            & (df['optim'] != 'SGD')]
    f = rough(pd.concat([df2, sgd]), 2)
    plt.figure(f.number)
    plt.title(r'All-CNN: CIFAR-10 (frac = 0.5)')
    plt.xlabel(r'wall-clock time (min)')
    plt.ylabel(r'top1 error ($\%$)')
    plt.xlim([0, 50])
    plt.ylim([5, 14])
    set_ticks(xt=[0, 10, 20, 30, 40, 50], yt=[5,8,11,14])

    plt.text(35, 5.44, r'$5.89$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=3)'])
    plt.text(35, 7.25, r'$6.51$', fontsize=fsz,
        verticalalignment='center', color=colors['Elastic-SGD (n=3)'])
    plt.text(40, 6.25, r'$6.15$', fontsize=fsz,
        verticalalignment='center', color=colors['SGD (full)'])

    if opt['s']:
        plt.savefig('../fig/allcnn_cifar10_half_valid.pdf', bbox_inches='tight')

    df2 = df[ (df['frac'] == 0.25)
            & (df['optim'] != 'Entropy-SGD')
            & (df['optim'] != 'SGD')]
    sgd['frac'].replace(0.5, 0.25, inplace=True)
    f = rough(pd.concat([df2, sgd]), 3)
    plt.figure(f.number)
    plt.title(r'All-CNN: CIFAR-10 (frac = 0.25)')
    plt.xlabel(r'wall-clock time (min)')
    plt.ylabel(r'top1 error ($\%$)')
    plt.xlim([0, 50])
    plt.ylim([5, 14])
    set_ticks(xt=[0, 10, 20, 30, 40, 50], yt=[5,8,11,14])

    plt.text(15, 6.5, r'$6.08$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=6)'])
    plt.text(15, 7.4, r'$6.8$', fontsize=fsz,
        verticalalignment='center', color=colors['Elastic-SGD (n=6)'])
    plt.text(35, 6.5, r'$6.15$', fontsize=fsz,
        verticalalignment='center', color=colors['SGD (full)'])

    if opt['s']:
        plt.savefig('../fig/allcnn_cifar10_fourth_valid.pdf', bbox_inches='tight')

def wrn_cifar10():
    f = rough(df[df['frac'] == 1], 1)
    plt.figure(f.number)
    plt.title(r'WRN-28-10: CIFAR-10')
    plt.xlabel(r'wall-clock time (min)')
    plt.ylabel(r'top1 error ($\%$)')
    plt.xlim([0, 400])
    plt.ylim([3,15])
    set_ticks(xt=[0, 100, 200, 300, 400], yt=[3,6,9,12,15])

    plt.text(80, 3.6, r'$3.77$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=8)'])
    plt.text(130, 3.6, r'$3.76$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=3)'])
    plt.text(250, 6.10, r'$4.38$', fontsize=fsz,
        verticalalignment='center', color=colors['Elastic-SGD (n=3)'])
    plt.text(350, 4.9, r'$4.23$', fontsize=fsz,
        verticalalignment='center', color=colors['Entropy-SGD'])
    plt.text(300, 4.9, r'$4.29$', fontsize=fsz,
        verticalalignment='center', color=colors['SGD'])

    if opt['s']:
        plt.savefig('../fig/wrn_cifar10_full_valid.pdf', bbox_inches='tight')

def wrn_cifar100():
    f = rough(df[df['frac'] == 1], 1)
    plt.figure(f.number)
    plt.title(r'WRN-28-10: CIFAR-100')
    plt.xlabel(r'wall-clock time (min)')
    plt.ylabel(r'top1 error ($\%$)')
    plt.xlim([0, 400])
    plt.ylim([15, 45])
    set_ticks(xt=[0, 100, 200, 300, 400], yt=[15,25,35,45])

    plt.text(350, 16.1, r'$17.64$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=3)'])
    plt.text(275, 16.1, r'$18.96$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=8)'])
    plt.text(350, 20.3, r'$19.05$', fontsize=fsz,
        verticalalignment='center', color=colors['Entropy-SGD'])
    plt.text(325, 22.5, r'$18.85$', fontsize=fsz,
        verticalalignment='center', color=colors['SGD'])

    if opt['s']:
        plt.savefig('../fig/wrn_cifar100_full_valid.pdf', bbox_inches='tight')

def wrn_svhn():
    f = rough(df[df['frac'] == 1], 1)
    plt.figure(f.number)
    plt.title(r'WRN-16-4: SVHN')
    plt.xlabel(r'wall-clock time (min)')
    plt.ylabel(r'top1 error ($\%$)')
    plt.xlim([0, 200])
    plt.ylim([15, 45])
    # set_ticks(xt=[0, 50, 100, 150, 200], yt=[15,25,35,45])


globals()[opt['m']]()