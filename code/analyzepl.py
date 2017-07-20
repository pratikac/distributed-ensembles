import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

import os, sys, glob2, pdb, re, json, argparse
import cPickle as pickle
from processlog import *

sns.set_style('ticks')
sns.set_color_codes()

parser = argparse.ArgumentParser(description='Analyze PL')
parser.add_argument('-i',
            help='input location', type=str,
            default='')
parser.add_argument('-f',
            help='for',
            action='store_true')
opt = vars(parser.parse_args())

if not opt['i'] == '':
    d = loaddir(opt['i'], expr='/*mnist*', force=opt['f'])

d['pl'] = d.fulldw**2/2/d.fullf
d['vardw'] = (d.fulldw - d.dw)**2

plt.figure(1)
plt.clf()
sns.tsplot(time='i',value='pl',data=d, unit='s')
