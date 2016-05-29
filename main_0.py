#!/usr/bin/env python3
from collections import namedtuple
from json import dumps
from numpy import *
from numpy.random import *
from scipy.io import *

################################################################################
# Constants and utility functions
################################################################################

eps = 1e-4

def aug(v):
    return concatenate([ones((1,) + v.shape[1:], float32), v])

def rect(v):
    return maximum(0, v)

def softmax(v):
    return maximum(eps, exp(v) / (eps + sum(exp(v), axis=0)))

def logis(v):
    return 1 / (1 + exp(-v))

def randnf(*dims):
    return float32(randn(*dims))

################################################################################
# Cascaded neural network definition
################################################################################

Net = namedtuple('Net', 'a b θ')

def train_net(x, y, arch=(), w_scale=1e-2, k_cpt=0, k_l2=0,
              s=(lambda t: t / 10), λ=(lambda t: 0.1), batch_size=64,
              n_epochs=200, log_progress=(lambda t, net: None)):
    n = x.shape[0:1] + arch
    k = y.shape[0]
    a = [w_scale * randnf(n[l + 1], n[l] + 1) for l in range(len(n) - 1)]
    b = [w_scale * randnf(k, n[l] + 1) for l in range(len(n))]
    θ = [1 for l in range(len(n) - 1)]
    net = Net(a, b, θ)
    for t in range(n_epochs):
        net = train_1_epoch(net, x, y, k_cpt, k_l2, s(t), λ(t), batch_size)
        print('========================================')
        print('Epoch', t)
        print('========================================')
        print('a[0]:', amin(net.a[0]), '–', amax(net.a[0]))
        print('a[1]:', amin(net.a[1]), '–', amax(net.a[1]))
        print('b[0]:', amin(net.b[0]), '–', amax(net.b[0]))
        print('b[1]:', amin(net.b[1]), '–', amax(net.b[1]))
        print('b[2]:', amin(net.b[2]), '–', amax(net.b[2]))
        print('θ[0]:', net.θ[0])
        print('θ[1]:', net.θ[1])
        log_progress(t, net)
    return net

def train_1_epoch(net, x, y, k_cpt, k_l2, s, λ, batch_size):
    order = permutation(x.shape[1])
    shuffled_x = take(x, order, axis=1)
    shuffled_y = take(y, order, axis=1)
    for i in range(0, x.shape[1] - batch_size, batch_size):
        x_i = shuffled_x[:, i : i + batch_size]
        y_i = shuffled_y[:, i : i + batch_size]
        net = train_1_batch(net, x_i, y_i, k_cpt, k_l2, s, λ)
    return net

def train_1_batch(net, x0, y, k_cpt, k_l2, s, λ):
    # Constant extraction
    n_layers = len(net.b)
    n_points = x0.shape[1]
    # Activity initialization
    x = n_layers * [None]
    p = n_layers * [None]
    d = n_layers * [None]
    c_cpt = n_layers * [None]
    c_err = n_layers * [None]
    c_tot = n_layers * [None]
    # Gradient initialization
    Δx = n_layers * [None]
    Δu = (n_layers - 1) * [None]
    Δv = n_layers * [None]
    Δp = n_layers * [None]
    Δd = n_layers * [None]
    Δc_tot = n_layers * [None]
    Δa = (n_layers - 1) * [None]
    Δb = n_layers * [None]
    Δθ = (n_layers - 1) * [None]
    # Activity propagation
    for l in range(n_layers):
        x[l] = x0 if l == 0 else rect(dot(net.a[l - 1], aug(x[l - 1])))
        p[l] = softmax(dot(net.b[l], aug(x[l])))
        d[l] = (1 if l == n_layers - 1 else
                logis(s * (amax(p[l], axis=0) - net.θ[l])))
        c_cpt[l] = 0 if l == 0 else k_cpt * net.a[l - 1].size
        c_err[l] = -sum(y * log(p[l]), axis=0)
    # Error analysis
    c_tot[-1] = c_cpt[-1] + c_err[-1]
    for l in reversed(range(n_layers - 1)):
        c_tot[l] = c_cpt[l] + d[l] * c_err[l] + (1 - d[l]) * c_tot[l + 1]
    # Error gradient propagation
    Δc_tot[0] = 1
    for l in range(1, n_layers):
        Δc_tot[l] = Δc_tot[l - 1] * (1 - d[l - 1])
    Δp[-1] = -Δc_tot[-1] * y / p[-1]
    Δv[-1] = p[-1] * (Δp[-1] - sum(Δp[-1] * p[-1], axis=0))
    Δb[-1] = dot(Δv[-1], aug(x[-1]).T) / n_points + 2 * k_l2 * net.b[-1]
    Δx[-1] = dot(net.b[-1].T[1:], Δv[-1])
    for l in reversed(range(n_layers - 1)):
        Δp[l] = Δc_tot[l] * d[l] * (
            (c_err[l] - c_tot[l + 1]) * s * (1 - d[l]) *
            float32(p[l] == argmax(p[l], axis=0)[newaxis]) -
            y / p[l])
        Δu[l] = Δx[l + 1] * float32(x[l + 1] > 0)
        Δv[l] = p[l] * (Δp[l] - sum(Δp[l] * p[l], axis=0))
        Δa[l] = dot(Δu[l], aug(x[l]).T) / n_points + 2 * k_l2 * net.a[l]
        Δb[l] = dot(Δv[l], aug(x[l]).T) / n_points + 2 * k_l2 * net.b[l]
        Δx[l] = dot(net.a[l].T[1:], Δu[l]) + dot(net.b[l].T[1:], Δv[l])
        Δd[l] = Δc_tot[l] * (c_err[l] - c_tot[l + 1])
        Δθ[l] = sum(Δd[l] * s * d[l] * (d[l] - 1)) / n_points
    # Parameter adjustment
    a = [net.a[l] - λ * Δa[l] for l in range(n_layers - 1)]
    b = [net.b[l] - λ * Δb[l] for l in range(n_layers)]
    θ = [clip(net.θ[l] - λ * Δθ[l], 0, 1) for l in range(n_layers - 1)]
    return Net(a, b, θ)

def predict(net, x, l=0):
    p = softmax(dot(net.b[l], aug(x)))
    if l == len(net.b) - 1 or amax(p) > net.θ[l]:
        return l, p
    else:
        next_x = rect(dot(net.a[l], aug(x)))
        return predict(net, next_x, l + 1)

def perf(net, x, y):
    right_answer_counts = zeros(len(net.b))
    wrong_answer_counts = zeros(len(net.b))
    for i in range(x.shape[1]):
        l, q = predict(net, x[:, i])
        if argmax(q) == argmax(y[:, i]):
            right_answer_counts[l] += 1
        else:
            wrong_answer_counts[l] += 1
    return right_answer_counts, wrong_answer_counts

################################################################################
# Data loading
################################################################################

categories = range(10)
mnist = loadmat('mnist.mat')

x_tr = hstack([mnist['train%i' % i] for i in categories])
x_ts = hstack([mnist['test%i' % i] for i in categories])
y_tr = hstack([
    tile(identity(10, float32)[i, :, newaxis], mnist['train%i' % i].shape[1])
    for i in categories
])
y_ts = hstack([
    tile(identity(10, float32)[i, :, newaxis], mnist['test%i' % i].shape[1])
    for i in categories
])

################################################################################
# Network training and evaluation
################################################################################

arch = (256, 256)
k_cpts = 1e-7 * arange(20)
k_l2 = 1e-4

results = []

for k_cpt in k_cpts:
    net = train_net(x_tr, y_tr, arch, k_cpt=k_cpt, k_l2=k_l2)
    op_counts = hstack([[0], cumsum(list(map(size, net.a)))])
    tr_rac, tr_wac = perf(net, x_tr, y_tr)
    ts_rac, ts_wac = perf(net, x_ts, y_ts)
    tr_acc = sum(tr_rac) / x_tr.shape[1]
    ts_acc = sum(ts_rac) / x_ts.shape[1]
    tr_moc = sum((tr_rac + tr_wac) * op_counts) / x_tr.shape[1]
    ts_moc = sum((ts_rac + ts_wac) * op_counts) / x_ts.shape[1]
    print('============================================================')
    print('k_cpt:', k_cpt)
    print('------------------------------------------------------------')
    print('training set performance:')
    print('  right answer counts:', tr_rac.tolist())
    print('  wrong answer counts:', tr_wac.tolist())
    print('  accuracy:', tr_acc)
    print('  mean op count:', tr_moc)
    print('test set performance:')
    print('  right answer counts:', ts_rac.tolist())
    print('  wrong answer counts:', ts_wac.tolist())
    print('  accuracy:', ts_acc)
    print('  mean op count:', ts_moc)
    results.append({
        'k_cpt': k_cpt.tolist(),
        'tr_rac': tr_rac.tolist(), 'tr_wac': tr_wac.tolist(),
        'tr_acc': tr_acc.tolist(), 'tr_moc': tr_moc.tolist(),
        'ts_rac': ts_rac.tolist(), 'ts_wac': ts_wac.tolist(),
        'ts_acc': ts_acc.tolist(), 'ts_moc': ts_moc.tolist()
    })

with open('results.json', 'w') as f:
    f.write(dumps(results, sort_keys=True, indent=2, separators=(',', ': ')))
