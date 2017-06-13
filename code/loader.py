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
            frac=1.0, weights=None):
        self.n = x.size(0)
        self.x, self.y = x.pin_memory(), y.pin_memory()
        self.num_classes = np.unique(self.y.numpy()).max() + 1

        if weights is None:
            self.weights = th.Tensor(self.n).fill_(1).double()
        else:
            self.weights = weights.clone().double()

        if train and frac < 1-1e-12:
            idx = th.randperm(self.n)
            self.x = th.index_select(self.x, 0, idx)
            self.y = th.index_select(self.y, 0, idx)
            self.n = int(self.n*frac)
            self.x, self.y = self.x[:self.n], self.y[:self.n]

            t1 = np.array(np.bincount(self.y.numpy(), minlength=self.num_classes))
            self.weights = th.from_numpy(float(self.n)/t1[self.y.numpy()]).double()

        self.b = batch_size
        self.idx = th.arange(0, self.b).long()
        self.train = train
        self.augment = augment
        self.sidx = 0

    def __next__(self):
        if self.train:
            self.idx.copy_(th.multinomial(self.weights, self.b, True))

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
    d1, d2 = datasets.MNIST('/local2/pratikac/mnist', train=True), \
            datasets.MNIST('/local2/pratikac/mnist', train=False)

    train = sampler_t(opt['b'], d1.train_data.view(-1,1,28,28).float(),
        d1.train_labels, augment=opt['augment'], frac=frac)
    train_full = sampler_t(opt['b'], d1.train_data.view(-1,1,28,28).float(),
        d1.train_labels, augment=opt['augment'], frac=1.0, train=False)
    val = sampler_t(opt['b'], d2.test_data.view(-1,1,28,28).float(),
        d2.test_labels, train=False)
    return train, val, val, train_full

def cifar10(opt):
    frac = opt.get('frac', 1.0)
    loc = '/local2/pratikac/cifar/'
    if 'resnet' in opt['m']:
        d1 = np.load(loc+'cifar10-train.npz')
        d2 = np.load(loc+'cifar10-test.npz')
    else:
        d1 = np.load(loc+'cifar10-train-proc.npz')
        d2 = np.load(loc+'cifar10-test-proc.npz')

    train = sampler_t(opt['b'], th.from_numpy(d1['data']),
                     th.from_numpy(d1['labels']), augment=opt['augment'], frac=frac)
    train_full = sampler_t(opt['b'], th.from_numpy(d1['data']),
                 th.from_numpy(d1['labels']), frac=1.0, train=False)
    val = sampler_t(opt['b'], th.from_numpy(d2['data']),
                     th.from_numpy(d2['labels']), train=False)
    return train, val, val, train_full

def cifar100(opt):
    frac = opt.get('frac', 1.0)
    loc = '/local2/pratikac/cifar/'
    if 'resnet' in opt['m']:
        d1 = np.load(loc+'cifar100-train.npz')
        d2 = np.load(loc+'cifar100-test.npz')
    else:
        d1 = np.load(loc+'cifar100-train-proc.npz')
        d2 = np.load(loc+'cifar100-test-proc.npz')

    train = sampler_t(opt['b'], th.from_numpy(d1['data']),
                     th.from_numpy(d1['labels']), augment=opt['augment'], frac=frac)
    train_full = sampler_t(opt['b'], th.from_numpy(d1['data']),
                     th.from_numpy(d1['labels']), frac=1.0, train=False)
    val = sampler_t(opt['b'], th.from_numpy(d2['data']),
                     th.from_numpy(d2['labels']), train=False)
    return train, val, val, train_full

def svhn(opt):
    frac = opt.get('frac', 1.0)
    loc = '/local2/pratikac/svhn'
    d1 = sio.loadmat(os.path.join(loc, 'train_32x32.mat'))
    d2 = sio.loadmat(os.path.join(loc, 'extra_32x32.mat'))
    d3 = sio.loadmat(os.path.join(loc, 'test_32x32.mat'))

    dt = {  'data': np.concatenate([d1['X'], d2['X']], axis=3),
            'labels': np.concatenate([d1['y'], d2['y']])-1}
    dv = {  'data': d3['X'],
            'labels': d3['y']-1}
    dt['data'] = np.transpose(dt['data'], (3,2,0,1))
    dv['data'] = np.transpose(dv['data'], (3,2,0,1))

    train = sampler_t(opt['b'], th.from_numpy(dt['data']).float()/255.,
                    th.from_numpy(dt['labels']).long(), augment=opt['augment'], frac=frac)
    train_full = sampler_t(opt['b'], th.from_numpy(dt['data']).float()/255.,
                    th.from_numpy(dt['labels']).long(), frac=1.0, train=False)
    val = sampler_t(opt['b'], th.from_numpy(dv['data']).float()/255.,
                     th.from_numpy(dv['labels']).long(), train=False)
    return train, val, val, train_full


class ReplacementSampler(object):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

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

    if opt['model'].startswith('vgg'):
        bsz = 64
        input_transform = [transforms.Scale(384)]
        normalize = [transforms.Lambda(lambda img: np.array(img) - np.array([123.68, 116.779, 103.939])),
            transforms.Lambda(lambda img: img[:,:,::-1]),    # RGB -> BGR
            transforms.Lambda(lambda pic:
                th.FloatTensor(pic).transpose(0,1).transpose(0,2).contiguous()
            )
        ]

    train_loader = th.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip()] + affine + normalize
            )),
        batch_size=bsz, shuffle=True,
        num_workers=nw, pin_memory=True)

    val_loader = None
    if not only_train:
        val_loader = th.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose(
                input_transform + [transforms.CenterCrop(224)] + affine + normalize)),
            batch_size=bsz, shuffle=False,
            num_workers=nw, pin_memory=True)

    return train_loader, val_loader