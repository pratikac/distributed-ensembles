import torch as th
import torchvision.cvtransforms as T
from torchvision import datasets, transforms
import torch.utils.data
import torchnet as tnt

import numpy as np
import os, sys, pdb, math, random
import cv2
import scipy.io as sio

class sampler_t:
    def __init__(self, batch_size, x,y, train=True, augment=False,
            frac=1.0, frac_start=0, balanced_sampling=False):
        self.n = x.size(0)
        idx = th.randperm(self.n)
        self.x = th.index_select(x, 0, idx)
        self.y = th.index_select(y, 0, idx)

        self.x, self.y = self.x.pin_memory(), self.y.pin_memory()
        self.balanced_sampling = balanced_sampling

        if train and frac < 1-1e-12:
            self.frac_start = frac_start
            self.frac = frac
            ns, ne = int(self.n*self.frac_start), int(self.n*(self.frac_start + self.frac))

            if ne <= self.n:
                self.x, self.y = self.x[ns:ne], self.y[ns:ne]
            else:
                ne = ne % self.n
                self.x, self.y =    th.cat((self.x[ns:], self.x[:ne])), \
                                    th.cat((self.y[ns:], self.y[:ne]))

            self.n = int(self.n*frac)

            if self.balanced_sampling:
                self.num_classes = np.unique(self.y.numpy()).max() + 1
                t1 = np.array(np.bincount(self.y.numpy(), minlength=self.num_classes))
                self.weights = th.from_numpy(float(self.n)/t1[self.y.numpy()]).double()

        self.b = batch_size
        self.idx = th.arange(0, self.b).long()
        self.train = train
        self.augment = augment
        self.sidx = 0

    def __next__(self):
        if self.train:
            if self.balanced_sampling:
                self.idx.copy_(th.multinomial(self.weights, self.b, True))
            else:
                self.idx.random_(0,self.n-1)

            x,y  = th.index_select(self.x, 0, self.idx), \
                    th.index_select(self.y, 0, self.idx)

            if self.augment:
                x = x.numpy().astype(np.float32)
                x = x.transpose(0,2,3,1)
                sz = x.shape[1]
                for i in xrange(self.b):
                    x[i] = T.RandomHorizontalFlip()(x[i])
                    res = T.Pad(4, cv2.BORDER_REFLECT)(x[i])
                    x[i] = T.RandomCrop(sz)(res)
                x = x.transpose(0,3,1,2)
                x = th.from_numpy(x)
        else:
            s = self.sidx
            e = min(s+self.b, self.n)

            self.idx = th.arange(s, e).long()
            self.sidx += self.b
            if self.sidx >= self.n:
                self.sidx = 0

            x,y  = th.index_select(self.x, 0, self.idx), \
                th.index_select(self.y, 0, self.idx)
        return x, y

    next = __next__

    def __iter__(self):
        return self

    def __len__(self):
        return int(math.ceil(self.n / float(self.b)))

def mnist(opt):
    frac = opt.get('frac', 1.0)
    frac_start = opt.get('frac_start', 0.0)

    d1, d2 = datasets.MNIST('/local2/pratikac/mnist', train=True), \
            datasets.MNIST('/local2/pratikac/mnist', train=False)

    train = sampler_t(opt['b'], d1.train_data.view(-1,1,28,28).float(),
        d1.train_labels, augment=opt['augment'], frac=frac, frac_start=frac_start)
    train_full = sampler_t(opt['b'], d1.train_data.view(-1,1,28,28).float(),
        d1.train_labels, augment=opt['augment'], frac=1.0, train=False)
    val = sampler_t(opt['b'], d2.test_data.view(-1,1,28,28).float(),
        d2.test_labels, train=False)
    return train, val, val, train_full

def cifar10(opt):
    frac = opt.get('frac', 1.0)
    frac_start = opt.get('frac_start', 0.0)
    loc = '/local2/pratikac/cifar/'

    if 'resnet' in opt['m']:
        d1 = np.load(loc+'cifar10-train.npz')
        d2 = np.load(loc+'cifar10-test.npz')
    else:
        d1 = np.load(loc+'cifar10-train-proc.npz')
        d2 = np.load(loc+'cifar10-test-proc.npz')

    train = sampler_t(opt['b'], th.from_numpy(d1['data']),
                     th.from_numpy(d1['labels']), augment=opt['augment'], frac=frac, frac_start=frac_start)
    train_full = sampler_t(opt['b'], th.from_numpy(d1['data']),
                 th.from_numpy(d1['labels']), frac=1.0, train=False)
    val = sampler_t(opt['b'], th.from_numpy(d2['data']),
                     th.from_numpy(d2['labels']), train=False)
    return train, val, val, train_full

def cifar100(opt):
    frac = opt.get('frac', 1.0)
    frac_start = opt.get('frac_start', 0.0)
    loc = '/local2/pratikac/cifar/'

    if 'resnet' in opt['m']:
        d1 = np.load(loc+'cifar100-train.npz')
        d2 = np.load(loc+'cifar100-test.npz')
    else:
        d1 = np.load(loc+'cifar100-train-proc.npz')
        d2 = np.load(loc+'cifar100-test-proc.npz')

    train = sampler_t(opt['b'], th.from_numpy(d1['data']),
                     th.from_numpy(d1['labels']), augment=opt['augment'], frac=frac, frac_start=frac_start)
    train_full = sampler_t(opt['b'], th.from_numpy(d1['data']),
                     th.from_numpy(d1['labels']), frac=1.0, train=False)
    val = sampler_t(opt['b'], th.from_numpy(d2['data']),
                     th.from_numpy(d2['labels']), train=False)
    return train, val, val, train_full

def svhn(opt):
    frac = opt.get('frac', 1.0)
    frac_start = opt.get('frac_start', 0.0)
    loc = '/local2/pratikac/svhn'

    d1 = sio.loadmat(os.path.join(loc, 'train_32x32.mat'))
    d2 = sio.loadmat(os.path.join(loc, 'extra_32x32.mat'))
    d3 = sio.loadmat(os.path.join(loc, 'test_32x32.mat'))

    dt = {  'data': np.concatenate([d1['X'], d2['X']], axis=3),
            'labels': np.concatenate([d1['y'], d2['y']])-1}
    dv = {  'data': d3['X'],
            'labels': d3['y']-1}
    dt['data'] = np.array(dt['data'], dtype=np.float32)
    dv['data'] = np.array(dv['data'], dtype=np.float32)

    dt['data'] = np.transpose(dt['data'], (3,2,0,1))
    dv['data'] = np.transpose(dv['data'], (3,2,0,1))

    mean = np.array([109.9, 109.7, 113.8])[None,:,None,None]
    std = np.array([50.1, 50.6, 50.9])[None,:,None,None]
    dt['data'] = (dt['data'] - mean)/std
    dv['data'] = (dv['data'] - mean)/std

    train = sampler_t(opt['b'], th.from_numpy(dt['data']).float(),
                    th.from_numpy(dt['labels']).long().squeeze(), augment=opt['augment'],
                    frac=frac, frac_start=frac_start)
    train_full = sampler_t(opt['b'], th.from_numpy(dt['data']).float(),
                    th.from_numpy(dt['labels']).long().squeeze(), frac=1.0, train=False)
    val = sampler_t(opt['b'], th.from_numpy(dv['data']).float(),
                     th.from_numpy(dv['labels']).long().squeeze(), train=False)
    return train, val, val, train_full

class ReplacementSampler(object):
    def __init__(self, data_source, num_samples, replacement=True):
        self.data_source = data_source
        self.weights = th.Tensor(len(data_source)).fill_(1)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(th.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return len(self.data_source)

def imagenet(opt, only_train=False):
    loc = '/local2/pratikac/imagenet'
    bsz, nw = opt['b'], 4

    traindir = os.path.join(loc, 'train')
    valdir = os.path.join(loc, 'val')

    input_transform = [transforms.Scale(256)]
    affine = []

    normalize = [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]

    if opt['m'].startswith('vgg'):
        bsz = 64
        input_transform = [transforms.Scale(384)]
        normalize = [transforms.Lambda(lambda img: np.array(img) - np.array([123.68, 116.779, 103.939])),
            transforms.Lambda(lambda img: img[:,:,::-1]),    # RGB -> BGR
            transforms.Lambda(lambda pic:
                th.FloatTensor(pic).transpose(0,1).transpose(0,2).contiguous()
            )
        ]

    train_folder = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip()] + affine + normalize))
    train_loader = th.utils.data.DataLoader(
        train_folder,
        batch_size=bsz, shuffle=True,
        num_workers=nw, pin_memory=True, sampler=ReplacementSampler(train_folder,
                                                    len(train_folder)*max(1,opt['L']),
                                                    replacement=True)
        )

    val_folder = datasets.ImageFolder(valdir, transforms.Compose(
            input_transform + [transforms.CenterCrop(224)] + affine + normalize))
    val_loader = th.utils.data.DataLoader(
        val_folder,
        batch_size=bsz, shuffle=False,
        num_workers=nw, pin_memory=True)

    return train_loader.__iter__(), val_loader.__iter__(), val_loader.__iter__(), train_loader.__iter__()

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