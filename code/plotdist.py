import pandas as pd
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import os, sys, glob, pdb, argparse
import cPickle as pickle
import seaborn as sns

sns.set_style('ticks')
sns.set_color_codes()

plt.ion()

dist = dict(sgd={}, esgd={})

dist['sgd']['per'] = \
np.array([[1.0, 0.17434936071229817, 0.18441392471255832, 0.1808696995930638, 0.17674880090262804, 0.18111243767610222],
[0.17434936071229817, 1.0, 0.17755346078148546, 0.17908863673145758, 0.18052005236270385, 0.17998675177832757],
[0.18441392471255832, 0.17755346078148546, 1.0, 0.18352942113876064, 0.18365592669531935, 0.18351874655730363],
[0.1808696995930638, 0.17908863673145758, 0.18352942113876064, 1.0, 0.18228261068946225, 0.17604470240182596],
[0.17674880090262804, 0.18052005236270385, 0.18365592669531935, 0.18228261068946225, 1.0, 0.18395642425281136],
[0.18111243767610222, 0.17998675177832757, 0.18351874655730363, 0.17604470240182596, 0.18395642425281136, 1.0]])

dist['esgd']['per'] = \
np.array([[1.0, 0.21631695868392545, 0.2190368412525279, 0.2257985907212584, 0.21939220623882888, 0.22891580532930997],
[0.21631695868392545, 1.0, 0.22065603845288434, 0.22372580188819896, 0.21627253336323604, 0.22116333385459286],
[0.2190368412525279, 0.22065603845288434,1.0, 0.22632583206623608, 0.22563445976511864, 0.2280974922675862],
[0.2257985907212584,0.22372580188819896,0.22632583206623608,1.0,0.2270392324734558,0.22461073014183178],
[0.21939220623882888,0.21627253336323604,0.22563445976511864,0.2270392324734558, 1.0,0.2247278653663592],
[0.22891580532930997,0.22116333385459286,0.2280974922675862,0.22461073014183178,0.2247278653663592,1.0]])

dist['esgd']['softmax'] = \
np.array([[ 0. ,0.03859545 , 0.03954068 , 0.03643564 , 0.03937737 , 0.03719942],
[ 0.03928439,  0.         , 0.0387048  , 0.036002   , 0.03631428 , 0.03590457],
[ 0.0396821 ,  0.03819139 , 0.         , 0.03683077 , 0.03801908 , 0.04059853],
[ 0.03649438 , 0.03565641 , 0.03659741 , 0.         , 0.03532869 , 0.03704991],
[ 0.04016383 , 0.03658656 , 0.03861252 , 0.03668762 , 0.         , 0.03613706],
[ 0.03760405 , 0.03552647 , 0.04103455 , 0.03796665 , 0.03560781 , 0.        ]])

dist['esgd']['softmax_dropout'] = \
np.array([[ 0.1281419 ,  0.15463095,  0.15781942,  0.15473528,  0.15515113, 0.15252503],
[ 0.15518377,  0.13352801,  0.15556815,  0.15335827,  0.15560943, 0.1573295 ],
[ 0.15641736,  0.15673664,  0.134758  ,  0.15925809,  0.15512521, 0.1573126 ],
[ 0.14833089,  0.1520756 ,  0.1542682 ,  0.12926223,  0.1498876 , 0.14913132],
[ 0.15162233,  0.15031675,  0.15143204,  0.15120159,  0.12795655, 0.15157727],
[ 0.15561783,  0.15838129,  0.15720345,  0.15897549,  0.1592684, 0.13328641]])

# loc = '/home/pratik/Dropbox/cs269data/permute/sgd/'
# dist['sgd']['unper'] = np.zeros((2,2))
# for i in xrange(2):
#     m1 = th.load(loc+'allcnn_s_'+str(i+42)+'.pz')['state_dict']
#     for j in xrange(2):
#         m2 = th.load(loc+'allcnn_s_'+str(j+42)+'.pz')['state_dict']
#         res = []
#         for k in m1:
#             if 'weight' in k:
#                 c = m1[k].size(0)
#                 w1, w2 = m1[k].cpu().numpy().reshape(c,-1), m2[k].cpu().numpy().reshape(c,-1)
#                 res.append(np.mean((w1*w2).sum(axis=0)/np.linalg.norm(w1, axis=0)/np.linalg.norm(w2, axis=0)))
#         dist['sgd']['unper'][i,j] = np.mean(res)

# loc = '/home/pratik/Dropbox/cs269data/permute/esgd/'
# dist['esgd']['unper'] = np.zeros((6,6))
# for i in xrange(6):
#     m1 = th.load(loc+'allcnn_s_'+str(i+42)+'.pz')['state_dict']
#     for j in xrange(6):
#         m2 = th.load(loc+'allcnn_s_'+str(j+42)+'.pz')['state_dict']
#         res = []
#         for k in m1:
#             if 'weight' in k:
#                 c = m1[k].size(0)
#                 w1, w2 = m1[k].cpu().numpy().reshape(c,-1), m2[k].cpu().numpy().reshape(c,-1)
#                 res.append(np.mean((w1*w2).sum(axis=0)/np.linalg.norm(w1, axis=0)/np.linalg.norm(w2, axis=0)))
#         dist['esgd']['unper'][i,j] = np.mean(res)


fsz = 18
plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=fsz)
plt.rc('xtick', labelsize=fsz)
plt.rc('ytick', labelsize=fsz)
plt.rc('legend', fontsize=fsz*0.8)
plt.rc('figure', titlesize=fsz)

a = np.around(dist['esgd']['per'], 3)
plt.figure(1, figsize=(6,6))
plt.clf()
ax = sns.heatmap(a, linewidths=0.5, fmt='.2f', cbar=False, annot=True, cmap='gist_yarg')
#ax.invert_yaxis()
plt.xticks([])
plt.yticks([])

plt.plot([0,0],[0,6],'k')
plt.plot([0,6],[0,0],'k')
plt.plot([6,6],[0,6],'k')
plt.plot([0,6],[6,6],'k')
plt.savefig('../fig/perdist.pdf', bbox_inches='tight')

a = np.around(dist['esgd']['softmax'], 3)
plt.figure(2, figsize=(6,6))
plt.clf()
ax = sns.heatmap(a, linewidths=0.5, fmt='.2f', cbar=False, annot=True, cmap='gist_yarg')
#ax.invert_yaxis()
plt.xticks([])
plt.yticks([])

plt.plot([0,0],[0,6],'k')
plt.plot([0,6],[0,0],'k')
plt.plot([6,6],[0,6],'k')
plt.plot([0,6],[6,6],'k')
plt.savefig('../fig/softmax_avg.pdf', bbox_inches='tight')

a = np.around(dist['esgd']['softmax_dropout'], 3)
plt.figure(3, figsize=(6,6))
plt.clf()
ax = sns.heatmap(a, linewidths=0.5, fmt='.2f', cbar=False, annot=True, cmap='gist_yarg')
#ax.invert_yaxis()
plt.xticks([])
plt.yticks([])

plt.plot([0,0],[0,6],'k')
plt.plot([0,6],[0,0],'k')
plt.plot([6,6],[0,6],'k')
plt.plot([0,6],[6,6],'k')
plt.savefig('../fig/softmax_avg_dropout.pdf', bbox_inches='tight')

