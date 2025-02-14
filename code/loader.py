import torch as th
import torchvision.cvtransforms as T
import torchvision.transforms as transforms
from torchvision import datasets
import torchnet as tnt
import torch.utils.data
import torchnet as tnt

import numpy as np
import os, sys, pdb, math, random
import cv2
import scipy.io as sio

home = '/home/'+os.environ['USER']

class InfDS(object):
    def __init__(self, d):
        self.d = d
        self.n = d['x'].size(0)

    def __getitem__(self, idx):
        i = idx % self.n
        return [self.d['x'][i], self.d['y'][i]]

    def __len__(self):
        return 2**20

def get_inf_iterator(d, transforms, bsz, nw=0, shuffle=True, pin_memory=True):
    ds = InfDS(d)
    ds = tnt.dataset.TransformDataset(ds, {0:transforms})
    return th.utils.data.DataLoader(ds, batch_size=bsz,
            num_workers=nw, shuffle=shuffle, pin_memory=pin_memory)

def get_iterator(d, transforms, bsz, nw=0, shuffle=True):
    ds = tnt.dataset.TensorDataset([d['x'], d['y']])
    ds = ds.transform({0:transforms})
    return ds.parallel(batch_size=bsz,
            num_workers=nw, shuffle=shuffle, pin_memory=True)

def shuffle_data(d):
    x, y = d['x'], d['y']
    n = x.size(0)
    idx = th.randperm(n)
    d['x'] = th.index_select(x, 0, idx)
    d['y'] = th.index_select(y, 0, idx)

def get_loaders(d, transforms, opt):
    if not opt['augment']:
        transforms = lambda x: x

    trf = get_iterator(d['train'], transforms, opt['b'], nw=opt['nw'], shuffle=True)
    trinff = get_inf_iterator(d['train'], transforms, opt['b'], nw=opt['nw'], shuffle=True)
    tv = get_iterator(d['val'], lambda x:x, opt['b'], nw=opt['nw'], shuffle=False)

    if opt['frac'] > 1-1e-12:
        return [dict(train=trinff,val=tv,test=tv,train_full=trf,
                idx=th.arange(0,d['train']['x'].size(0))) for i in range(opt['n'])]
    else:
        n = opt['n']
        N = d['train']['x'].size(0)
        tr = []
        idxs = []
        for i in range(n):
            fs = (i / float(n)) % 1.0
            ns, ne = int(N*fs), int(N*(fs+opt['frac']))
            x, y = d['train']['x'], d['train']['y']

            if ne <= N:
                idxs.append(th.arange(ns,ne).long())
                xy = {'x': x[ns:ne], 'y': y[ns:ne]}
            else:
                ne = ne % N
                idxs.append(th.cat((th.arange(ns,N), th.arange(0,ne))).long())
                xy = {  'x': th.cat((x[ns:], x[:ne])),
                        'y': th.cat((y[ns:], y[:ne]))}
            tr.append(get_inf_iterator(xy, transforms, opt['b'], nw=0, shuffle=True))
        return [dict(train=tr[i],val=tv,test=tv,train_full=trf,idx=idxs[i]) for i in range(opt['n'])]

def mnist(opt):
    loc = home + '/local2/pratikac/mnist'
    d1, d2 = datasets.MNIST(loc, train=True), datasets.MNIST(loc, train=False)

    d = {'train': {'x': d1.train_data.view(-1,1,28,28).float(), 'y': d1.train_labels},
        'val': {'x': d2.test_data.view(-1,1,28,28).float(), 'y': d2.test_labels}}

    shuffle_data(d['train'])
    return d, lambda x: x

def cifar_helper(opt, s):
    loc = home + '/local2/pratikac/cifar/'
    if 'resnet' in opt['m'] or 'densenet' in opt['m']:
        d1 = np.load(loc+s+'-train.npz')
        d2 = np.load(loc+s+'-test.npz')
    else:
        d1 = np.load(loc+s+'-train-proc.npz')
        d2 = np.load(loc+s+'-test-proc.npz')

    d = {'train': {'x': th.from_numpy(d1['data']), 'y': th.from_numpy(d1['labels'])},
        'val': {'x': th.from_numpy(d2['data']), 'y': th.from_numpy(d2['labels'])}}
    shuffle_data(d['train'])

    sz = d['train']['x'].size(3)
    augment = tnt.transform.compose([
        lambda x: x.numpy().astype(np.float32),
        lambda x: x.transpose(1,2,0),
        T.RandomHorizontalFlip(),
        T.Pad(4, 2),
        T.RandomCrop(sz),
        lambda x: x.transpose(2,0,1),
        th.from_numpy])

    return d, augment

def cifar10(opt):
    return cifar_helper(opt, 'cifar10')

def cifar100(opt):
    return cifar_helper(opt, 'cifar100')

def svhn(opt):
    loc = home + '/local2/pratikac/svhn/'

    d1 = sio.loadmat(loc + 'train_32x32.mat')
    d2 = sio.loadmat(loc + 'extra_32x32.mat')
    d3 = sio.loadmat(loc + 'test_32x32.mat')

    d = {'train': { 'x': np.concatenate([d1['X'],d2['X']], axis=3).astype(np.float32),
                    'y': np.concatenate([d1['y'],d2['y']])-1},
        'val': {'x': d3['X'].astype(np.float32),
                'y': d3['y']-1}}

    mean = np.array([109.9, 109.7, 113.8])[None,:,None,None]
    std = np.array([50.1, 50.6, 50.9])[None,:,None,None]

    for k in d:
        d[k]['x'] = np.transpose(d[k]['x'], (3,2,0,1))
        d[k]['x'] = (d[k]['x'] - mean)/std

        d[k]['x'] = th.from_numpy(d[k]['x'])
        d[k]['y'] = th.from_numpy(d[k]['y'])

    shuffle_data(d['train'])

    sz = d['train']['x'].size(3)
    augment = tnt.transform.compose([
        lambda x: x.numpy().astype(np.float32),
        lambda x: x.transpose(1,2,0),
        T.RandomHorizontalFlip(),
        T.Pad(4, 2),
        T.RandomCrop(sz),
        lambda x: x.transpose(2,0,1),
        th.from_numpy])

    return d, lambda x: x

def imagenet(opt, only_train=False):
    loc = home + '/local2/pratikac/imagenet'
    bsz, nw = opt['b'], 4

    traindir = os.path.join(loc, 'train')
    valdir = os.path.join(loc, 'val')

    input_transform = [transforms.Scale(256)]

    normalize = [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]


    train_folder = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip()] + normalize))
    train_loader = th.utils.data.DataLoader(
        train_folder,
        batch_size=bsz, shuffle=True,
        num_workers=nw, pin_memory=True)

    val_folder = datasets.ImageFolder(valdir, transforms.Compose(
            input_transform + [transforms.CenterCrop(224)] + normalize))
    val_loader = th.utils.data.DataLoader(
        val_folder,
        batch_size=bsz, shuffle=False,
        num_workers=nw, pin_memory=True)

    ids = th.arange(0, len(train_loader)).long()

    return [dict(train=train_loader,
                val=val_loader,
                test=val_loader,
                train_full=train_loader,
                idx=ids) for i in range(opt['n'])]

# PTB
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self):
        path = home + '/local2/pratikac/ptb'
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r') as f:
            ids = th.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

def ptb(opt):
    c = Corpus()

    def batchify(d, bsz):
        nb = d.size(0) // bsz
        d = d.narrow(0, 0, nb*bsz)
        d = d.view(bsz, -1).t().contiguous()
        return d

    def get_batch(src, i):
        l = min(opt['T'], len(src)-1-i)
        return src[i:i+l], src[i+1:i+1+l].view(-1)

    r = {'train': batchify(c.train, opt['b']),
        'val': batchify(c.valid, opt['b']),
        'test': batchify(c.test, opt['b'])}
    return  c, r, get_batch


class FederatedSampler(object):
    def __init__(self, opt, d, idxs, transforms):
        self.transforms = transforms
        self.n = opt['n']
        self.N = d['train']['x'].size(0)
        self.b = opt['b']
        self.d = d

        self.idxs = th.Tensor(self.n, self.N).zero_()
        for i, idx in enumerate(idxs):
            for ii in idx:
                self.idxs[i][ii] = 1.0

        self.sidx = th.arange(0, self.b).long()

    def next(self, i):
        self.sidx.copy_(th.multinomial(self.idxs[i], self.b, True))
        x = th.index_select(self.d['train']['x'], 0, self.sidx)
        y = th.index_select(self.d['train']['y'], 0, self.sidx).squeeze()
        for i in range(self.b):
            x[i] = self.transforms(x[i])
        return x,y

def get_federated_loaders(d, transforms, opt):
    if not opt['augment']:
        transforms = lambda x: x

    trf = get_iterator(d['train'], transforms, opt['b'], nw=opt['nw'], shuffle=True)
    tv = get_iterator(d['val'], lambda x:x, opt['b'], nw=opt['nw'], shuffle=False)

    n = opt['n']
    N = d['train']['x'].size(0)
    idxs = []
    for i in range(n):
        fs = (i / float(n)) % 1.0
        ns, ne = int(N*fs), int(N*(fs+opt['frac']))
        if ne <= N:
            idxs.append(th.arange(ns,ne).long())
        else:
            ne = ne % N
            idxs.append(th.cat((th.arange(ns,N), th.arange(0,ne))).long())

    return dict(train=FederatedSampler(opt, d, idxs, transforms), val=tv, train_full=trf)
