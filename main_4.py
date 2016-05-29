#!/usr/bin/env python3
import itertools
import os

os.environ['THEANO_FLAGS'] = (
    'device=gpu,floatX=float32,cast_policy=numpy+floatX,' +
    'enable_initial_driver_test=False,warn_float64=raise')

import numpy as np
import numpy.random as rand
import scipy.io as io
import theano as th
import theano.tensor as ts
import theano.tensor.nnet as nn

################################################################################
# Transformation Definitions
################################################################################

class ReLuTF:
    def __init__(self, n_in, n_out, w_scale):
        self.w = th.shared(np.float32(w_scale * rand.randn(n_in, n_out)))
        self.b = th.shared(np.float32(w_scale * rand.randn(n_out)))

    def params(self):
        return [self.w, self.b]

    def link(self, x, k_cpt, k_l2):
        self.x = nn.relu(ts.dot(x, self.w) + self.b)
        self.c_cpt = k_cpt * ts.cast(self.w.size, 'float32')
        # self.l_l2 = k_l2 * (ts.sum(ts.sqr(self.w)) + ts.sum(ts.sqr(self.b)))
        self.l_l2 = k_l2 * ts.sum(ts.sqr(self.w))

class IdentityTF:
    def __init__(self):
        pass

    def params(self):
        return []

    def link(self, x, k_cpt, k_l2):
        self.x = x
        self.c_cpt = 0
        self.l_l2 = 0

################################################################################
# Neural Decision Tree Definition
################################################################################

class Layer:
    def __init__(self, n_in, n_lab, w_scale, tf, children):
        n_act = n_lab + len(children)
        self.w = th.shared(np.float32(w_scale * rand.randn(n_in, n_act)))
        self.b = th.shared(np.float32(w_scale * rand.randn(n_act)))
        self.tf = tf
        self.children = children

    def params(self):
        return list(itertools.chain(
            [self.w, self.b], self.tf.params(),
            *[c.params() for c in self.children]))

    def link(self, p, x, y, k_cpt, k_l2, ϵ):
        # infer activity shape
        n_lab = self.w.shape[1].eval() - len(self.children)
        n_act = self.w.shape[1].eval()

        # link to the transformation
        self.tf.link(x, k_cpt, k_l2)

        # propagate activity
        c_act_est = ts.dot(self.tf.x, self.w) + self.b
        i_fav_act = ts.argmin(c_act_est, axis=1, keepdims=True)
        δ_fav_act = ts.eq(ts.arange(n_act), i_fav_act)
        p_act = ϵ / np.float32(n_act - 1) + (1 - ϵ - ϵ / np.float32(n_act - 1)) * δ_fav_act

        # link recursively
        for i, child in enumerate(self.children):
            child.link(p * p_act[:, n_lab+i, None], self.tf.x, y, k_cpt, k_l2, ϵ)

        # perform error analysis
        c_err = ts.cast(ts.neq(ts.arange(n_lab), y), 'float32')
        c_del = [ch.tf.c_cpt + ch.c_fav_act for ch in self.children]
        c_act = ts.concatenate([c_err] + c_del, axis=1)
        self.c_fav_act = ts.sum(δ_fav_act * c_act, axis=1, keepdims=True)

        # compute the local loss
        l_err = ts.sqr(c_act_est - c_act)
        # l_l2 = k_l2 * ts.sum(ts.sqr(self.w)) + ts.sum(ts.sqr(self.b)) + self.tf.l_l2
        l_l2 = k_l2 * ts.sum(ts.sqr(self.w)) + self.tf.l_l2
        self.l_lay = p * ts.sum(p_act * (l_err + l_l2), axis=1, keepdims=True)

        # compute global loss
        self.mean_path_len = p + sum(ch.mean_path_len for ch in self.children)
        self.l_tot = (self.l_lay + sum(ch.l_lay for ch in self.children)) / self.mean_path_len

        # classify
        self.y_est = ts.concatenate(
            [i_fav_act] +
            [ts.eq(i_fav_act, n_lab + i) * ch.y_est + ts.neq(i_fav_act, n_lab + i) * -1
             for i, ch in enumerate(self.children)],
            axis=1)

class Net:
    def __init__(self, root):
        x = ts.fmatrix()
        y = ts.icol()
        k_cpt = ts.fscalar()
        k_l2 = ts.fscalar()
        ϵ = ts.fscalar()
        λ = ts.fscalar()
        root.link(1, x, y, k_cpt, k_l2, ϵ)
        l_avg = ts.mean(root.l_tot)
        self.train = th.function([x, y, k_cpt, k_l2, ϵ, λ], l_avg, updates=[
            (p, p - λ * ts.grad(l_avg, p))
            for p in root.params()
        ])
        self.classify = th.function([x], root.y_est)

################################################################################
# Data Loading
################################################################################

mnist = io.loadmat('mnist.mat')

x_tr = np.vstack([mnist['train%i' % i].T for i in range(10)])
x_ts = np.vstack([mnist['test%i' % i].T for i in range(10)])

y_tr = np.int32(np.vstack([
    i * np.ones((mnist['train%i' % i].shape[1], 1))
    for i in range(10)]))
y_ts = np.int32(np.vstack([
    i * np.ones((mnist['test%i' % i].shape[1], 1))
    for i in range(10)]))

################################################################################
# Network Training and Evaluation
################################################################################

def train_1_epoch(net, x, y, k_cpt, k_l2, ϵ, λ, batch_size):
    order = rand.permutation(x.shape[0])
    x = np.take(x, order, axis=0)
    y = np.take(y, order, axis=0)
    losses = []
    for i in range(0, x.shape[0] - batch_size, batch_size):
        losses.append(net.train(x[i:i+batch_size], y[i:i+batch_size], k_cpt, k_l2, ϵ, λ))
    print('mean_loss:', np.mean(losses))

k_cpt = np.float32(-1e-5)
k_l2 = np.float32(1e-6)

w_scale = np.float32(1e-3)
ϵ = lambda t: np.float32(0.5)
λ = lambda t: np.float32(1e-3)
batch_size = 512
n_epochs = 5001

net = Net(
    Layer(784, 10, w_scale, IdentityTF(), [
        Layer(256, 10, w_scale, ReLuTF(784, 256, w_scale), [
            Layer(256, 10, w_scale, ReLuTF(256, 256, w_scale), [])
        ])
    ])
)

for t in range(n_epochs):
    if t % 10 == 0:
        n_lab = np.max(y_ts) + 1
        y_est = net.classify(x_ts)
        δ_cor = y_est == y_ts
        δ_inc = (y_est >= 0) * (y_est < n_lab) * (y_est != y_ts)
        n_cor = np.sum(δ_cor, axis=0).tolist()
        n_inc = np.sum(δ_inc, axis=0).tolist()
        acc = np.sum(n_cor) / x_ts.shape[0]
        print('============================================================')
        print('Epoch', t)
        print('============================================================')
        print('n_correct:', n_cor)
        print('n_incorrect:', n_inc)
        print('accuracy:', acc)
    train_1_epoch(net, x_tr, y_tr, k_cpt, k_l2, ϵ(t), λ(t), batch_size)
