# Modified from From Taco Cohen's Github: https://github.com/tscohen/gconv_experiments

import os, argparse, pdb
import numpy as np
import cPickle as pickle

parser = argparse.ArgumentParser(description='Process CIFAR-10/100')
parser.add_argument('-d','--data',   help='Directory containing batches',
        type=str, required=True)
parser.add_argument('-o','--output',  help='Output',
        type=str, required=True)
parser.add_argument('-n','--name',  help='cifar10 | cifar100',
        type=str, required=True)
opt = parser.parse_args()

class PCA(object):
    def __init__(self, D, n_components):
        self.n_components = n_components
        self.U, self.S, self.m = self.fit(D, n_components)

    def fit(self, D, n_components):
        """
        The computation works as follows:
        The covariance is C = 1/(n-1) * D * D.T
        The eigendecomp of C is: C = V Sigma V.T
        Let Y = 1/sqrt(n-1) * D
        Let U S V = svd(Y),
        Then the columns of U are the eigenvectors of:
        Y * Y.T = C
        And the singular values S are the sqrts of the eigenvalues of C
        We can apply PCA by multiplying by U.T
        """

        # We require scaled, zero-mean data to SVD,
        # But we don't want to copy or modify user data
        m = np.mean(D, axis=1)[:, np.newaxis]
        D -= m
        D *= 1.0 / np.sqrt(D.shape[1] - 1)
        U, S, V = np.linalg.svd(D, full_matrices=False)
        D *= np.sqrt(D.shape[1] - 1)
        D += m
        return U[:, :n_components], S[:n_components], m

    def transform(self, D, whiten=False, ZCA=False,
                  regularizer=10 ** (-5)):
        """
        We want to whiten, which can be done by multiplying by Sigma^(-1/2) U.T
        Any orthogonal transformation of this is also white,
        and when ZCA=True we choose:
         U Sigma^(-1/2) U.T
        """
        if whiten:
            # Compute Sigma^(-1/2) = S^-1,
            # with smoothing for numerical stability
            Sinv = 1.0 / (self.S + regularizer)

            if ZCA:
                # The ZCA whitening matrix
                W = np.dot(self.U,
                           np.dot(np.diag(Sinv),
                                  self.U.T))
            else:
                # The whitening matrix
                W = np.dot(np.diag(Sinv), self.U.T)

        else:
            W = self.U.T

        # Transform
        return np.dot(W, D - self.m)

cifar_mean = np.array([125.3, 123.0, 113.9])[None,:,None,None]
cifar_std = np.array([63.0, 62.1, 66.7])[None,:,None,None]

def proc():
    def _load_batch(fn):        
        fo = open(fn, 'rb')
        d = pickle.load(fo)
        fo.close()
        if opt.name == 'cifar10':
            return d['data'].reshape(-1, 3, 32, 32), d['labels']
        else:
            return d['data'].reshape(-1, 3, 32, 32), d['fine_labels']

    def normalize(data, eps=1e-8):
        data -= data.mean(axis=(1, 2, 3), keepdims=True)
        std = np.sqrt(data.var(axis=(1, 2, 3), ddof=1, keepdims=True))
        std[std < eps] = 1.
        data /= std
        return data

    print '[Loading]'
    if opt.name == 'cifar10':
        train_fns = [os.path.join(opt.data, 'data_batch_' + str(i)) for i in range(1, 6)]
        train_batches = [_load_batch(fn) for fn in train_fns]
        test_batch = _load_batch(os.path.join(opt.data, 'test_batch'))
    elif opt.name == 'cifar100':
        train_fns = [os.path.join(opt.data, 'train')]
        train_batches = [_load_batch(fn) for fn in train_fns]
        test_batch = _load_batch(os.path.join(opt.data, 'test'))

    otx = np.vstack([train_batches[i][0] for i in range(len(train_batches))]).astype('float32')
    ty = np.vstack([train_batches[i][1] for i in range(len(train_batches))]).flatten()
    ovx, vy = test_batch[0].astype('float32'), test_batch[1]

    wtx = (otx - cifar_mean)/cifar_std
    wvx = (ovx - cifar_mean)/cifar_std
    print '[Saving basic mean-std]'
    np.savez(os.path.join(opt.output, opt.name+'-train.npz'), data=wtx.astype('float32'), labels=ty)
    np.savez(os.path.join(opt.output, opt.name+'-test.npz'), data=wvx.astype('float32'), labels=vy)

    tx, vx = normalize(otx), normalize(ovx)
    txf, vxf = tx.reshape(tx.shape[0], -1).T, vx.reshape(vx.shape[0], -1).T

    print '[PCA]'
    pca = PCA(D=txf, n_components=txf.shape[1])    
    tx = pca.transform(D=txf, whiten=True, ZCA=True).T.reshape(tx.shape)
    vx = pca.transform(D=vxf, whiten=True, ZCA=True).T.reshape(vx.shape)

    print '[Saving processed]'
    np.savez(os.path.join(opt.output, opt.name+'-train-proc.npz'), data=tx.astype('float32'), labels=ty)
    np.savez(os.path.join(opt.output, opt.name+'-test-proc.npz'), data=vx.astype('float32'), labels=vy)

    print '[Finished]'

proc()