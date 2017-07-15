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
    tv = get_iterator(d['val'], lambda x:x, opt['b'], nw=opt['nw'], shuffle=False)
    if opt['frac'] > 1-1e-12:
        return [dict(train=trf,val=tv,test=tv,train_full=trf) for i in xrange(opt['n'])]
    else:
        n = d['train']['x'].size(0)
        tr = []
        for i in xrange(n):
            fs = (i % float(n)) % 1
            ns, ne = int(n*fs), int(n*(fs+opt['frac']))
            x, y = d['train']['x'], d['train']['y']
            if ne <= n:
                xy = {'x': x[ns:ne], 'y': y[ns:ne]}
            else:
                ne = ne % n
                xy = {  'x': th.cat((x[ns:], x[:ne])),
                        'y': th.cat((y[ns:], y[:ne]))}
            tr.append(get_iterator(xy, transforms, opt['b'], nw=opt['nw'], shuffle=True))
        return [dict(train=tr[i],val=tv,test=tv,train_full=trf) for i in xrange(opt['n'])]

def mnist(opt):
    loc = '/local2/pratikac/mnist'
    d1, d2 = datasets.MNIST(loc, train=True), datasets.MNIST(loc, train=False)

    d = {'train': {'x': d1.train_data.view(-1,1,28,28).float(), 'y': d1.train_labels},
        'val': {'x': d2.test_data.view(-1,1,28,28).float(), 'y': d2.test_labels}}

    shuffle_data(d['train'])
    return d, lambda x: x

def cifar10(opt):
    loc = '/local2/pratikac/cifar/'
    if 'resnet' in opt['m']:
        d1 = np.load(loc+'cifar10-train.npz')
        d2 = np.load(loc+'cifar10-test.npz')
    else:
        d1 = np.load(loc+'cifar10-train-proc.npz')
        d2 = np.load(loc+'cifar10-test-proc.npz')

    d = {'train': {'x': th.from_numpy(d1['data']), 'y': th.from_numpy(d1['labels'])},
        'val': {'x': th.from_numpy(d2['data']), 'y': th.from_numpy(d2['labels'])}}
    shuffle_data(d['train'])

    sz = d['train']['x'].size(3)
    augment = tnt.transform.compose([
        lambda x: x.numpy().astype(np.float32),
        lambda x: x.transpose(1,2,0),
        T.RandomHorizontalFlip(),
        T.Pad(4, cv2.BORDER_REFLECT),
        T.RandomCrop(sz),
        lambda x: x.transpose(2,0,1),
        th.from_numpy])

    return d, augment

def cifar100(opt):
    loc = '/local2/pratikac/cifar/'
    if 'resnet' in opt['m']:
        d1 = np.load(loc+'cifar100-train.npz')
        d2 = np.load(loc+'cifar100-test.npz')
    else:
        d1 = np.load(loc+'cifar100-train-proc.npz')
        d2 = np.load(loc+'cifar100-test-proc.npz')

    d = {'train': {'x': th.from_numpy(d1['data']), 'y': th.from_numpy(d1['labels'])},
        'val': {'x': th.from_numpy(d2['data']), 'y': th.from_numpy(d2['labels'])}}
    shuffle_data(d['train'])

    sz = d['train']['x'].size(3)
    augment = tnt.transform.compose([
        lambda x: x.numpy().astype(np.float32),
        lambda x: x.transpose(1,2,0),
        T.RandomHorizontalFlip(),
        T.Pad(4, cv2.BORDER_REFLECT),
        T.RandomCrop(sz),
        lambda x: x.transpose(2,0,1),
        th.from_numpy])

    return d, lambda x: x

def svhn(opt):
    loc = '/local2/pratikac/svhn/'

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
        T.Pad(4, cv2.BORDER_REFLECT),
        T.RandomCrop(sz),
        lambda x: x.transpose(2,0,1),
        th.from_numpy])

    return d, lambda x: x

def imagenet(opt, only_train=False):
    loc = '/local2/pratikac/imagenet'
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

    return train_loader, val_loader, val_loader, train_loader

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
        path = '/local2/pratikac/ptb'
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
