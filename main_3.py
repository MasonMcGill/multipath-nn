#!/usr/bin/env python3
import itertools
import os

os.environ['THEANO_FLAGS'] = (
    'device=gpu,floatX=float32,allow_gc=False,nvcc.fastmath=True,' +
    'enable_initial_driver_test=False,warn_float64=warn')

import numpy as np
import numpy.random as rand
import scipy.io as io
import theano as th
import theano.tensor as ts
import theano.tensor.nnet as nn

################################################################################
# Neural Decision Tree Definition
################################################################################

class Counter:
    def __init__(self):
        self.value = 0

class ReLuTF:
    def __init__(self, n_in, n_out, w_scale):
        self.w = th.shared(np.float32(w_scale * rand.randn(n_in, n_out)))
        self.b = th.shared(np.float32(w_scale * rand.randn(n_out)))

    def params(self):
        return [self.w, self.b]

    def link(self, x, k_cpt, k_l2):
        self.x = nn.relu(ts.dot(x, self.w) + self.b)
        self.c_cpt = k_cpt * ts.cast(self.w.size, 'float32')
        self.loss_l2 = k_l2 * ts.sum(ts.sqr(self.w)) + ts.sum(ts.sqr(self.b))

class IdentityTF:
    def __init__(self):
        pass

    def params(self):
        return []

    def link(self, x, k_cpt, k_l2):
        self.x = x
        self.c_cpt = 0
        self.loss_l2 = 0

class Layer:
    def __init__(self, n, k, w_scale, tf, children):
        self.k = k
        self.w = th.shared(np.float32(w_scale * rand.randn(n, k + len(children))))
        self.b = th.shared(np.float32(w_scale * rand.randn(k + len(children))))
        self.tf = tf
        self.children = children

    def params(self):
        return list(itertools.chain(
            [self.w, self.b], self.tf.params(),
            *[c.params() for c in self.children]))

    def link(self, p, x, y, k_cpt, k_l2, ϵ, l_counter):
        l = l_counter.value
        l_counter.value += 1

        self.tf.link(x, k_cpt, k_l2)

        n_act = self.k + len(self.children)
        c_act_est = ts.dot(self.tf.x, self.w) + self.b
        i_fav_act = ts.argmin(c_act_est, axis=1, keepdims=True)
        δ_fav_act = ts.eq(ts.arange(n_act), i_fav_act)
        p_act = ϵ / (n_act - 1) + (1 - ϵ - ϵ / (n_act - 1)) * δ_fav_act

        for i, child in enumerate(self.children):
            child.link(p * p_act[:, self.k+i, np.newaxis], self.tf.x,
                       y, k_cpt, k_l2, ϵ, l_counter)

        c_err = ts.cast(ts.neq(ts.arange(self.k), y), 'float32')
        c_act = (
            self.tf.c_cpt +
            ts.concatenate(
                [c_err] + [child.c_fav_act for child in self.children],
                axis=1))
        self.c_fav_act = ts.sum(δ_fav_act * c_act, axis=1, keepdims=True)

        loss_err = ts.sqr(c_act_est - c_act)
        loss_l2 = k_l2 * ts.sum(ts.sqr(self.w)) + ts.sum(ts.sqr(self.b)) + self.tf.loss_l2
        self.loss_lay = p * ts.sum(p_act * (loss_err + loss_l2), axis=1, keepdims=True)
        self.mean_path_len = p + sum(child.mean_path_len for child in self.children)
        self.loss = (self.loss_lay + sum(child.loss for child in self.children)) / self.mean_path_len

        y_est_act = ts.concatenate(
            [ts.arange(self.k, dtype='float32') * ts.ones((x.shape[0], 1))] +
            [c.y_est for c in self.children],
            axis=1)
        l_cl_act = ts.concatenate(
            [l * ts.ones((x.shape[0], self.k))] +
            [c.l_cl for c in self.children],
            axis=1)
        self.y_est = ts.sum(δ_fav_act * y_est_act, axis=1, keepdims=True)
        self.l_cl = ts.sum(δ_fav_act * l_cl_act, axis=1, keepdims=True)

class Net:
    def __init__(self, root):
        x = ts.fmatrix()
        y = ts.icol()
        k_cpt = ts.fscalar()
        k_l2 = ts.fscalar()
        ϵ = ts.fscalar()
        λ = ts.fscalar()
        root.link(1, x, y, k_cpt, k_l2, ϵ, Counter())
        mean_loss = ts.mean(root.loss)
        self.train = th.function([x, y, k_cpt, k_l2, ϵ, λ], mean_loss, updates=[
            (p, p - λ * ts.grad(mean_loss, p))
            for p in root.params()
        ])
        self.classify = th.function([x], [root.y_est, root.l_cl])

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

def train_1_epoch(net, x0, y, k_cpt, k_l2, ϵ, λ, batch_size):
    order = rand.permutation(x0.shape[0])
    x0 = np.take(x0, order, axis=0)
    y = np.take(y, order, axis=0)
    losses = []
    for i in range(0, x0.shape[0] - batch_size, batch_size):
        losses.append(net.train(x0[i:i+batch_size], y[i:i+batch_size], k_cpt, k_l2, ϵ, λ))
    print('mean_cost:', np.mean(losses))

def get_perf(net, x, y):
    right_answer_counts = np.zeros(3)
    wrong_answer_counts = np.zeros(3)
    y_est, l_cl = net.classify(x)
    try:
        for i in range(x.shape[0]):
            if y_est[i] == y[i]:
                right_answer_counts[l_cl[i, 0]] += 1
            else:
                wrong_answer_counts[l_cl[i, 0]] += 1
    except Exception as e:
        print('offending index:', i, l_cl[i, 0])
        print('shapes:', x.shape, l_cl.shape)
        exit()
    return right_answer_counts, wrong_answer_counts

def log_state(net, t):
    tr_rac, tr_wac = get_perf(net, x_tr, y_tr)
    ts_rac, ts_wac = get_perf(net, x_ts, y_ts)
    tr_acc = np.sum(tr_rac) / x_tr.shape[0]
    ts_acc = np.sum(ts_rac) / x_ts.shape[0]
    print('============================================================')
    print('Epoch', t)
    print('============================================================')
    print('training set performance:')
    print('  right answer counts:', tr_rac.tolist())
    print('  wrong answer counts:', tr_wac.tolist())
    print('  accuracy:', tr_acc)
    print('test set performance:')
    print('  right answer counts:', ts_rac.tolist())
    print('  wrong answer counts:', ts_wac.tolist())
    print('  accuracy:', ts_acc)

k_cpt = np.float32(5e-7)
k_l2 = np.float32(1e-4)

w_scale = np.float32(1e-3)
ϵ = lambda t: np.float32(0.5)
λ = lambda t: np.float32(1e-2)
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
        log_state(net, t)
    train_1_epoch(net, x_tr, y_tr, k_cpt, k_l2, ϵ(t), λ(t), batch_size)
