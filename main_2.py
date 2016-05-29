#!/usr/bin/env python3
import os
os.environ['THEANO_FLAGS'] = (
    'device=cpu,allow_gc=False,floatX=float32,' +
    'enable_initial_driver_test=False,warn_float64=warn')

import numpy as np
import numpy.random as rand
import scipy.io as io
import theano as th
import theano.tensor as ts
import theano.tensor.nnet as nn

################################################################################
# Cascaded Neural Network Definition
################################################################################

class Net:
    def __init__(self, n, k, w_scale):
        self._init_params(n, k, w_scale)
        self._def_train()
        self._def_classify()

    def _init_params(self, n, k, w_scale):
        self.b = len(n) * [None]
        self.β = len(n) * [None]
        self.a = (len(n) - 1) * [None]
        self.α = (len(n) - 1) * [None]
        for l in range(len(n)):
            self.b[-1] = th.shared(np.float32(w_scale * rand.randn(n[-1], k)))
            self.β[-1] = th.shared(np.float32(w_scale * rand.randn(k)))
        for l in range(len(n) - 1):
            self.a[l] = th.shared(np.float32(w_scale * rand.randn(*n[l:l+2])))
            self.α[l] = th.shared(np.float32(w_scale * rand.randn(n[l+1])))
            self.b[l] = th.shared(np.float32(w_scale * rand.randn(n[l], k + 1)))
            self.β[l] = th.shared(np.float32(w_scale * rand.randn(k + 1)))

    def _def_classify(self):
        x0 = ts.fmatrix()
        k = self.b[-1].shape[1]
        x = lambda l: (
            x0 if l == 0 else
            nn.relu(ts.dot(x(l - 1), self.a[l-1]) + self.α[l-1]))
        c_act_est = lambda l: ts.dot(x(l), self.b[l]) + self.β[l]
        d = lambda l: ts.argmin(c_act_est(l), axis=1, keepdims=True)
        y_est = lambda l: (
            d(l) if l == len(self.b) - 1 else
            ts.switch(d(l) < k, d(l), y_est(l + 1)))
        l_cl = lambda l: (
            l if l == len(self.b) - 1 else
            ts.switch(d(l) < k, l, l_cl(l + 1)))
        self.classify = th.function([x0], [y_est(0), l_cl(0)])

    def _def_train(self):
        x0 = ts.fmatrix()
        y = ts.icol()
        k_cpt = ts.fscalar()
        k_l2 = ts.fscalar()
        ϵ = ts.fscalar()
        μ = ts.fscalar()
        λ = ts.fscalar()
        k = self.b[-1].shape[1]

        x = len(self.b) * [None]
        c_act_est = len(self.b) * [None]
        p_act_cond = len(self.b) * [None]
        p_act = len(self.b) * [None]
        δ_best = len(self.b) * [None]
        c_err = len(self.b) * [None]
        c_cpt = len(self.b) * [None]
        c_act = len(self.b) * [None]
        c_best = len(self.b) * [None]
        loss_l2 = len(self.b) * [None]

        for l in range(len(self.b)):
            x[l] = (x0 if l == 0 else
                    nn.relu(ts.dot(x[l-1], self.a[l-1]) + self.α[l-1]))
            c_act_est[l] = ts.dot(x[l], self.b[l]) + self.β[l]
            n_act = ts.cast(self.b[l].shape[1], 'float32')
            i_best = ts.argmin(c_act_est[l], axis=1, keepdims=True)
            δ_best[l] = ts.cast(ts.eq(ts.arange(n_act), i_best), 'float32')
            p_act_cond[l] = (ϵ / (n_act - 1) +
                             (1 - ϵ - ϵ / (n_act - 1)) * δ_best[l])
            p_act[l] = p_act_cond[l] * (1 if l == 0 else p_act[l-1][:, k, None])

        for l in reversed(range(len(self.b))):
            c_err[l] = ts.cast(ts.neq(ts.arange(k), y), 'float32')
            c_cpt[l] = k_cpt * ts.cast(self.a[l-1].size, 'float32')
            c_act[l] = (
                c_err[l] if l == len(self.b) - 1 else
                ts.concatenate((c_err[l], c_cpt[l+1] + c_best[l+1]), axis=1))
            c_best[l] = ts.sum(δ_best[l] * c_act[l], axis=1, keepdims=True)

        for l in range(len(self.b)):
            loss_l2[l] = k_l2 * (
                ((0 if l == 0 else
                  ts.sum(ts.sqr(self.a[l-1])) + ts.sum(ts.sqr(self.α[l-1]))) +
                 ts.sum(ts.sqr(self.b[l])) + ts.sum(ts.sqr(self.β[l]))))

        layer_losses = [
            ts.sum(p_act[l] * (ts.sqr(c_act_est[l] - c_act[l]) + loss_l2[l])) /
            ts.sum(p_act[l])
            for l in range(len(self.b))]
        loss = sum(layer_losses)

        self.train = th.function([x0, y, k_cpt, k_l2, ϵ, λ], loss, updates=[
            (w, w - λ * ts.grad(loss, w))
            for w in self.b + self.β + self.a + self.α
        ])

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

# x_tr = np.float32(np.vstack([
#     io.loadmat('cifar-10/data_batch_1.mat')['data'],
#     io.loadmat('cifar-10/data_batch_2.mat')['data'],
#     io.loadmat('cifar-10/data_batch_3.mat')['data'],
#     io.loadmat('cifar-10/data_batch_4.mat')['data'],
#     io.loadmat('cifar-10/data_batch_5.mat')['data']
# ]) / 255)
# y_tr = np.int32(np.vstack([
#     io.loadmat('cifar-10/data_batch_1.mat')['labels'],
#     io.loadmat('cifar-10/data_batch_2.mat')['labels'],
#     io.loadmat('cifar-10/data_batch_3.mat')['labels'],
#     io.loadmat('cifar-10/data_batch_4.mat')['labels'],
#     io.loadmat('cifar-10/data_batch_5.mat')['labels']
# ]))
#
# x_ts = np.float32(io.loadmat('cifar-10/test_batch.mat')['data'] / 255)
# y_ts = np.int32(io.loadmat('cifar-10/test_batch.mat')['labels'])

# x_tr = np.take(x_tr, np.where(y_tr < 2)[0], axis=0)
# y_tr = np.take(y_tr, np.where(y_tr < 2)[0], axis=0)
# x_ts = np.take(x_ts, np.where(y_ts < 2)[0], axis=0)
# y_ts = np.take(y_ts, np.where(y_ts < 2)[0], axis=0)

################################################################################
# Network Training and Evaluation
################################################################################

def train_1_epoch(net, x0, y, k_cpt, k_l2, ϵ, λ, batch_size):
    order = rand.permutation(x0.shape[0])
    x0 = np.take(x0, order, axis=0)
    y = np.take(y, order, axis=0)
    for i in range(0, x0.shape[0] - batch_size, batch_size):
        net.train(x0[i:i+batch_size], y[i:i+batch_size], k_cpt, k_l2, ϵ, λ)

def get_perf(net, x0, y):
    right_answer_counts = np.zeros(len(net.b))
    wrong_answer_counts = np.zeros(len(net.b))
    y_est, l_cl = net.classify(x0)
    for i in range(x0.shape[0]):
        if y_est[i] == y[i]:
            right_answer_counts[l_cl[i]] += 1
        else:
            wrong_answer_counts[l_cl[i]] += 1
    return right_answer_counts, wrong_answer_counts

def log_state(net, t):
    tr_rac, tr_wac = get_perf(net, x_tr, y_tr)
    ts_rac, ts_wac = get_perf(net, x_ts, y_ts)
    tr_acc = np.sum(tr_rac) / x_tr.shape[0]
    ts_acc = np.sum(ts_rac) / x_ts.shape[0]
    print('============================================================')
    print('Epoch', t)
    print('============================================================')
    for l in range(len(net.a)):
        a_l = net.a[l].get_value()
        print('a[%i]: min=%f max=%f mean_abs=%f' %
              (l, np.min(a_l), np.max(a_l), np.mean(np.abs(a_l))))
    for l in range(len(net.b)):
        b_l = net.b[l].get_value()
        print('b[%i]: min=%f max=%f mean_abs=%f' %
              (l, np.min(b_l), np.max(b_l), np.mean(np.abs(b_l))))
    print('------------------------------------------------------------')
    print('training set performance:')
    print('  right answer counts:', tr_rac.tolist())
    print('  wrong answer counts:', tr_wac.tolist())
    print('  accuracy:', tr_acc)
    print('test set performance:')
    print('  right answer counts:', ts_rac.tolist())
    print('  wrong answer counts:', ts_wac.tolist())
    print('  accuracy:', ts_acc)

n = (x_tr.shape[1], 256, 256)
k = np.max(y_tr) + 1
k_cpt = np.float32(5e-7)
k_l2 = np.float32(1e-4)

w_scale = np.float32(1e-3)
ϵ = lambda t: np.float32(0.5)
λ = lambda t: np.float32(1e-2)
batch_size = 512
n_epochs = 5001

net = Net(n, k, w_scale)
for t in range(n_epochs):
    if t % 10 == 0:
        log_state(net, t)
    train_1_epoch(net, x_tr, y_tr, k_cpt, k_l2, ϵ(t), λ(t), batch_size)
