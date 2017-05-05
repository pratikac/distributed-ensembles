import torch as th
from torchvision import datasets, transforms
import torch.utils.data

import numpy as np
import os, sys, pdb, math, random
import cv2

class sampler_t:
    def __init__(self, batch_size, x,y, train=True, augment=False):
        self.n = x.size(0)
        self.x, self.y = x,y
        self.b = batch_size
        self.idx = th.range(0, self.b-1).long()
        self.train = train
        self.augment = augment
        self.sidx = 0

    def __next__(self):
        if self.train:
            self.idx.random_(0,self.n-1)

            x,y  = th.index_select(self.x, 0, self.idx), \
                    th.index_select(self.y, 0, self.idx)

            if self.augment:
                p = 4
                r = np.random.randint(0, 1, self.b)
                x = x.numpy().transpose(0,2,3,1)
                sz = x.shape[1]
                for i in xrange(self.b):
                    if r[i] == 1:
                        x[i] = np.fliplr(x[i])

                    # pad using reflect to preserve color
                    res = cv2.copyMakeBorder(x[i], p,p,p,p,
                            borderType=cv2.BORDER_REFLECT, value=0)

                    szx, szy = res.shape[:2]
                    sz1, sz2 = random.randint(0, szx-sz), random.randint(0, szy-sz)
                    res = res[sz1:sz1+sz, sz2:sz2+sz]
                    res = res.reshape(x[i].shape)

                x = x.transpose(0,3,1,2)
                x = th.from_numpy(x)
        else:
            s = self.sidx
            e = min(s+self.b-1, self.n-1)

            self.idx = th.range(s, e).long()
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
    d1, d2 = datasets.MNIST('/local2/pratikac/mnist', train=True), \
            datasets.MNIST('/local2/pratikac/mnist', train=False)

    train = sampler_t(opt['b'], d1.train_data.view(-1,1,28,28).float(),
        d1.train_labels, augment=opt['augment'])
    val = sampler_t(opt['b'], d2.test_data.view(-1,1,28,28).float(),
        d2.test_labels, train=False)
    return train, val, val

def rotmnist(opt):
    loc = '/local2/pratikac/rotmnist/'
    d1 = np.load(loc+'mnist_all_rotation_normalized_float_train_valid.npy')
    d2 = np.load(loc+'mnist_all_rotation_normalized_float_test.npy')

    train = sampler_t(opt['b'], th.from_numpy(d1[:,:-1]).float().view(-1,1,28,28)/255.,
            th.from_numpy(d1[:,-1]).long(), augment=opt['augment'])
    val = sampler_t(opt['b'], th.from_numpy(d2[:,:-1]).float().view(-1,1,28,28)/255.,
            th.from_numpy(d2[:,-1]).long(), train=False)

    return train, val, val

def cifar10(opt):
    loc = '/local2/pratikac/cifar/'
    if 'resnet' in opt['m']:
        d1 = np.load(loc+'cifar10-train.npz')
        d2 = np.load(loc+'cifar10-test.npz')
    else:
        d1 = np.load(loc+'cifar10-train-proc.npz')
        d2 = np.load(loc+'cifar10-test-proc.npz')

    train = sampler_t(opt['b'], th.from_numpy(d1['data']),
                     th.from_numpy(d1['labels']), augment=opt['augment'])
    val = sampler_t(opt['b'], th.from_numpy(d2['data']),
                     th.from_numpy(d2['labels']), train=False)
    return train, val, val

def cifar100(opt):
    loc = '/local2/pratikac/cifar/'
    if 'resnet' in opt['m']:
        d1 = np.load(loc+'cifar100-train.npz')
        d2 = np.load(loc+'cifar100-test.npz')
    else:
        d1 = np.load(loc+'cifar100-train-proc.npz')
        d2 = np.load(loc+'cifar100-test-proc.npz')

    train = sampler_t(opt['b'], th.from_numpy(d1['data']),
                     th.from_numpy(d1['labels']), augment=opt['augment'])
    val = sampler_t(opt['b'], th.from_numpy(d2['data']),
                     th.from_numpy(d2['labels']), train=False)
    return train, val, val

def imagenet(opt, only_train=False):
    loc = '/local2/pratikac/imagenet'
    bsz, nw = opt['b'], opt['t']

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
    b = opt['b']

    def batchify(d):
        nb = d.size(0) // b
        d = d.narrow(0, 0, nb*b)
        d = d.view(b, -1).t().contiguous()
        return d

    def get_batch(src, i, volatile=False):
        l = min(opt['T'], len(src)-1-i)
        return src.narrow(0,i,l), src.narrow(0,i+1, l).view(-1)

    r = {'train': batchify(c.train),
         'valid': batchify(c.valid),
         'test': batchify(c.test)}
    return  c, r, get_batch