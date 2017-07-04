import numpy as np
import numpy.random as npr
from scipy.signal import correlate, convolve, gaussian, exponential
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
import os, sys, argparse

plt.ion()
sns.set_style('white')

npr.seed(42)

s1, s2, s3 = 0.1, 0.2, 0.1
rv1 = multivariate_normal([0.75,0.5], np.eye(2)*s1)
rv2 = multivariate_normal([0.25,0.1], np.eye(2)*s2)
rv3 = multivariate_normal([0.1,0.8], np.eye(2)*s3)

x,y = np.mgrid[0:1:0.01, 0:1:0.01]
pos = np.empty(x.shape + (2,))
pos[:,:,0], pos[:,:,1] = x, y

f = -(rv1.pdf(pos) + rv2.pdf(pos) + rv3.pdf(pos))/3.

plt.figure(1)
plt.clf()
sns.interactplot(x.reshape(-1), y.reshape(-1), f.reshape(-1), levels=20,
        filled=True, scatter_kws={'marker':'.', 'markersize':0.01}, rasterized=True,
        colorbar=False)
plt.xticks([])
plt.yticks([])
plt.savefig('esgd.pdf', bbox_inches='tight')