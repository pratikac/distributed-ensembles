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
    fsz = 18
    if opt['s']:
        fsz = 24
    plt.rc('font', size=fsz)
    plt.rc('axes', titlesize=fsz)
    plt.rc('axes', labelsize=fsz)
    plt.rc('xtick', labelsize=fsz)
    plt.rc('ytick', labelsize=fsz)
    plt.rc('legend', fontsize=fsz*0.8)
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
df['ee'] = df['e']*df['L']*df['frac']

df['optim'] = 'GD'
df.loc[(df['L'] == 1) & (df['n'] == 1), 'optim'] = 'SGD'
df.loc[(df['L'] != 1) & (df['n'] == 1), 'optim'] = 'Entropy-SGD'
for ni in np.unique(df.n):
    if ni != 1:
        df.loc[(df['L'] != 1) & (df['n'] == ni), 'optim'] = 'Parle (n=%d)'%(ni)
        df.loc[(df['L'] == 1) & (df['n'] == ni), 'optim'] = 'Elastic-SGD (n=%d)'%(ni)

if opt['m'] != 'lenet':
    eff = 0.6
    df.loc[(df['optim'] == 'SGD') | (df['optim'] == 'Entropy-SGD'), 'ee'] *= eff

sgd = df[(df['optim']=='SGD') & (df['frac'] == 1.0)].copy()
sgd.replace({'SGD':'SGD (full)'}, inplace=True)

def rough(d, idx=1):
    dc = d.copy()
    dc = dc.filter(items=['f','top1','s','ee','optim','n','val','train'])

    # val
    fig = plt.figure(idx, figsize=(8,7))
    plt.clf()

    dv = dc[(dc['val'] == True)]
    sns.tsplot(time='ee',value='top1',data=dv,
                unit='s',condition='optim', color=colors)
    sns.tsplot(time='ee',value='top1',
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
    # sns.tsplot(time='ee',value='top1',data=dt,
    #             unit='s',condition='optim', color=colors,
    #             legend=False)
    # sns.tsplot(time='ee',value='top1',
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
    plt.title('LeNet: MNIST')
    plt.xlim([0, 100])
    plt.ylim([0.4, 1.0])
    plt.xlabel('Effective epochs')
    set_ticks(yt=[0.4, 0.6, 0.8, 1.0])

    plt.text(65, 0.415, r'$0.44 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=6)'])
    plt.text(85, 0.415, r'$0.48 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['Elastic-SGD (n=6)'])
    plt.text(85, 0.57, r'$0.49 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['Entropy-SGD'])
    plt.text(70, 0.57, r'$0.50 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['SGD'])
    if opt['s']:
        plt.savefig('../fig/lenet_full_valid.pdf', bbox_inches='tight')

def allcnn_cifar10():
    f = rough(df[df['frac'] == 1], 1)
    plt.figure(f.number)
    plt.title('All-CNN: CIFAR-10 (full)')
    plt.xlabel('Effective epochs')
    plt.xlim([0, 250])
    plt.ylim([5, 14])
    set_ticks(xt=[0, 50, 100, 150, 200, 250], yt=[5,8,11,14])

    plt.text(225, 5.8, r'$5.18 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=3)'])
    plt.text(150, 5.18, r'$5.76 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['Entropy-SGD'])
    plt.text(100, 7, r'$6.15 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['SGD'])

    if opt['s']:
        plt.savefig('../fig/allcnn_cifar10_full_valid.pdf', bbox_inches='tight')

    sgd['frac'].replace(1.0, 0.5, inplace=True)
    df2 = df[ (df['frac'] == 0.5)
            & (df['optim'] != 'Entropy-SGD')
            & (df['optim'] != 'SGD')]
    f = rough(pd.concat([df2, sgd]), 2)
    plt.figure(f.number)
    plt.title('All-CNN: CIFAR-10 (frac = 0.5)')
    plt.xlabel('Effective epochs')
    plt.xlim([0, 150])
    plt.ylim([5, 15])
    set_ticks(xt=[0, 50, 100, 150], yt=[5,10,15])

    plt.text(92, 5.44, r'$5.89 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=3)'])
    plt.text(80, 8.0, r'$6.51 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['Elastic-SGD (n=3)'])
    plt.text(110, 6.8, r'$6.15 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['SGD (full)'])

    if opt['s']:
        plt.savefig('../fig/allcnn_cifar10_half_valid.pdf', bbox_inches='tight')

    df2 = df[ (df['frac'] == 0.25)
            & (df['optim'] != 'Entropy-SGD')
            & (df['optim'] != 'SGD')]
    sgd['frac'].replace(0.5, 0.25, inplace=True)
    f = rough(pd.concat([df2, sgd]), 3)
    plt.figure(f.number)
    plt.title('All-CNN: CIFAR-10 (frac = 0.25)')
    plt.xlabel('Effective epochs')
    plt.xlim([0, 100])
    plt.ylim([6, 18])
    set_ticks(xt=[0, 25, 50, 75, 100], yt=[6,9,12,15,18])

    plt.text(55, 6.5, r'$6.08 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=6)'])
    plt.text(40, 7.4, r'$6.8 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['Elastic-SGD (n=6)'])
    plt.text(85, 8, r'$6.15 \%$', fontsize=fsz,
        verticalalignment='center', color=colors['SGD (full)'])

    if opt['s']:
        plt.savefig('../fig/allcnn_cifar10_fourth_valid.pdf', bbox_inches='tight')

def wrn_cifar10():
    f = rough(df[df['frac'] == 1], 1)
    plt.figure(f.number)
    plt.title('WRN-28-10: CIFAR-10')
    plt.xlabel('Effective epochs')
    plt.xlim([0, 150])
    plt.ylim([3,15])
    set_ticks(xt=[0, 50, 100, 150], yt=[3,6,9,12,15])

    plt.text(25, 3.6, r'$3.77\%$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=8)'])
    plt.text(130, 3.6, r'$3.76\%$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=3)'])
    plt.text(75, 6.10, r'$4.38\%$', fontsize=fsz,
        verticalalignment='center', color=colors['Elastic-SGD (n=3)'])
    plt.text(130, 4.9, r'$4.23\%$', fontsize=fsz,
        verticalalignment='center', color=colors['Entropy-SGD'])
    plt.text(100, 4.9, r'$4.29\%$', fontsize=fsz,
        verticalalignment='center', color=colors['SGD'])

    if opt['s']:
        plt.savefig('../fig/wrn_cifar10_full_valid.pdf', bbox_inches='tight')

def wrn_cifar100():
    f = rough(df[df['frac'] == 1], 1)
    plt.figure(f.number)
    plt.title('WRN-28-10: CIFAR-100')
    plt.xlabel('Effective epochs')
    plt.xlim([0, 150])
    plt.ylim([15, 45])
    set_ticks(xt=[0, 50, 100, 150], yt=[15,25,35,45])

    plt.text(75, 16.1, r'$17.64\%$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=3)'])
    plt.text(73, 22, r'$18.96\%$', fontsize=fsz,
        verticalalignment='center', color=colors['Parle (n=8)'])
    plt.text(125, 20.3, r'$19.05\%$', fontsize=fsz,
        verticalalignment='center', color=colors['Entropy-SGD'])
    plt.text(105, 22.5, r'$18.85\%$', fontsize=fsz,
        verticalalignment='center', color=colors['SGD'])

    if opt['s']:
        plt.savefig('../fig/wrn_cifar100_full_valid.pdf', bbox_inches='tight')

def wrn_svhn():
    f = rough(df[df['frac'] == 1], 1)
    plt.figure(f.number)
    plt.title('WRN-16-4: SVHN')
    plt.xlabel('epochs x L')
    plt.xlim([0, 200])
    plt.ylim([15, 45])
    # set_ticks(xt=[0, 50, 100, 150, 200], yt=[15,25,35,45])

globals()[opt['m']]()