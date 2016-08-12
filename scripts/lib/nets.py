from abc import ABCMeta, abstractmethod
from functools import reduce
from types import SimpleNamespace as Namespace

import numpy as np
import tensorflow as tf

from lib.layers import BatchNorm, Chain, LinTrans, Rect

################################################################################
# Optimization
################################################################################

def minimize_expected(net, cost, optimizer):
    lr_scales = {
        param: 1 / tf.sqrt(tf.reduce_mean(tf.square(ℓ.p_tr)))
        for ℓ in net.layers for param in [
            *vars(ℓ.params).values(),
            *vars(ℓ.router.params).values()]}
    grads = optimizer.compute_gradients(cost)
    scaled_grads = [(lr_scales[p] * g, p) for g, p in grads]
    return optimizer.apply_gradients(scaled_grads)

################################################################################
# Root Network Class
################################################################################

class Net(metaclass=ABCMeta):
    def __init__(self, x0_shape, y_shape, root):
        self.x0 = tf.placeholder(tf.float32, (None,) + x0_shape)
        self.y = tf.placeholder(tf.float32, (None,) + y_shape)
        self.mode = tf.placeholder_with_default('ev', ())
        self.root = root
        def link(ℓ, x):
            ℓ.link(Namespace(x=x, y=self.y, mode=self.mode))
            for s in ℓ.sinks:
                link(s, ℓ.x)
        link(self.root, self.x0)

    @property
    def layers(self):
        def all_in_tree(layer):
            yield layer
            for sink in layer.sinks:
                yield from all_in_tree(sink)
        yield from all_in_tree(self.root)

    @property
    def leaves(self):
        return (ℓ for ℓ in self.layers if len(ℓ.sinks) == 0)

    def train(self, x0, y, hypers):
        pass

################################################################################
# Statically-Routed Networks
################################################################################

class SRNet(Net):
    def __init__(self, x0_shape, y_shape, optimizer, root):
        super().__init__(x0_shape, y_shape, root)
        for ℓ in self.layers:
            ℓ.p_ev = tf.ones((tf.shape(ℓ.x)[0],))
        c_tr = sum(ℓ.c_err + ℓ.c_mod for ℓ in self.layers)
        self._train_op = optimizer.minimize(tf.reduce_mean(c_tr))

    def train(self, x0, y, hypers):
        self._train_op.run({self.x0: x0, self.y: y, self.mode: 'tr', **hypers})

################################################################################
# Decision Smoothing Networks
################################################################################

def route_ds_stat(ℓ, p_tr, p_ev, opts):
    ℓ.p_tr = p_tr
    ℓ.p_ev = p_ev
    ℓ.router = Chain({}, [])
    ℓ.router.link(Namespace(x=ℓ.x, mode=opts.mode))
    for s in ℓ.sinks:
        route_ds(s, ℓ.p_tr, ℓ.p_ev, opts)

def route_ds_dyn(ℓ, p_tr, p_ev, opts):
    ℓ.p_tr = p_tr
    ℓ.p_ev = p_ev
    ℓ.router = Chain({},
        sum(([LinTrans(dict(n_chan=n, k_l2=opts.k_l2)), BatchNorm({}), Rect({})]
             for n in opts.arch), [])
        + [LinTrans(dict(n_chan=len(ℓ.sinks), k_l2=opts.k_l2))])
    ℓ.router.link(Namespace(x=ℓ.x, mode=opts.mode))
    π_tr = (
        opts.ϵ / len(ℓ.sinks)
        + (1 - opts.ϵ) * tf.nn.softmax(ℓ.router.x))
    π_ev = tf.to_float(tf.equal(
        tf.expand_dims(tf.to_int32(tf.argmax(ℓ.router.x, 1)), 1),
        tf.range(len(ℓ.sinks))))
    for i, s in enumerate(ℓ.sinks):
        route_ds(s, ℓ.p_tr * π_tr[:, i], ℓ.p_ev * π_ev[:, i], opts)

def route_ds(ℓ, p_tr, p_ev, opts):
    if len(ℓ.sinks) < 2: route_ds_stat(ℓ, p_tr, p_ev, opts)
    else: route_ds_dyn(ℓ, p_tr, p_ev, opts)

class DSNet(Net):
    def __init__(self, x0_shape, y_shape, arch, k_l2, optimizer, root):
        super().__init__(x0_shape, y_shape, root)
        self.k_cpt = tf.placeholder_with_default(0.0, ())
        self.ϵ = tf.placeholder_with_default(0.01, ())
        n_pts = tf.shape(self.x0)[0]
        route_ds(self.root, tf.ones((n_pts,)), tf.ones((n_pts,)),
                 Namespace(arch=arch, k_l2=k_l2, ϵ=self.ϵ, mode=self.mode))
        c_err = sum(ℓ.p_tr * ℓ.c_err for ℓ in self.layers)
        c_cpt = sum(ℓ.p_tr * self.k_cpt * ℓ.n_ops for ℓ in self.layers)
        c_mod = sum(tf.stop_gradient(ℓ.p_tr) * (ℓ.c_mod + ℓ.router.c_mod)
                    for ℓ in self.layers)
        c_tr = c_err + c_cpt + c_mod
        self._train_op = minimize_expected(
            self, tf.reduce_mean(c_tr), optimizer)

    def train(self, x0, y, hypers):
        self._train_op.run({self.x0: x0, self.y: y, self.mode: 'tr', **hypers})

################################################################################
# Cost Regression Networks
################################################################################

def route_cr_stat(ℓ, p_tr, p_ev, opts):
    ℓ.p_tr = p_tr
    ℓ.p_ev = p_ev
    ℓ.router = Chain({}, [])
    ℓ.router.link(Namespace(x=ℓ.x, mode=opts.mode))
    for s in ℓ.sinks:
        route_cr(s, ℓ.p_tr, ℓ.p_ev, opts)
    ℓ.c_cre = 0.0
    ℓ.c_ev = (
        ℓ.c_err + opts.k_cpt * ℓ.n_ops
        + sum(s.c_ev for s in ℓ.sinks))

def route_cr_dyn(ℓ, p_tr, p_ev, opts):
    ℓ.p_tr = p_tr
    ℓ.p_ev = p_ev
    ℓ.router = Chain({},
        sum(([LinTrans(dict(n_chan=n, k_l2=opts.k_l2)), BatchNorm({}), Rect({})]
             for n in opts.arch), [])
        + [LinTrans(dict(n_chan=len(ℓ.sinks), k_l2=opts.k_l2))])
    ℓ.router.link(Namespace(x=ℓ.x, mode=opts.mode))
    π_ev = tf.to_float(tf.equal(
        tf.expand_dims(tf.to_int32(tf.argmin(ℓ.router.x, 1)), 1),
        tf.range(len(ℓ.sinks))))
    π_tr = opts.ϵ / len(ℓ.sinks) + (1 - opts.ϵ) * π_ev
    for i, s in enumerate(ℓ.sinks):
        route_cr(s, ℓ.p_tr * π_tr[:, i], ℓ.p_ev * π_ev[:, i], opts)
    ℓ.c_cre = opts.k_cre * sum(
        π_tr[:, i] * tf.square(tf.stop_gradient(s.c_ev) - ℓ.router.x[:, i])
        for i, s in enumerate(ℓ.sinks))
    ℓ.c_ev = (
        ℓ.c_err + opts.k_cpt * ℓ.n_ops
        + sum(π_ev[:, i] * s.c_ev
              for i, s in enumerate(ℓ.sinks)))

def route_cr(ℓ, p_tr, p_ev, opts):
    if len(ℓ.sinks) < 2: route_cr_stat(ℓ, p_tr, p_ev, opts)
    else: route_cr_dyn(ℓ, p_tr, p_ev, opts)

class CRNet(Net):
    def __init__(self, x0_shape, y_shape, arch, k_l2, optimizer, root):
        super().__init__(x0_shape, y_shape, root)
        self.k_cpt = tf.placeholder_with_default(0.0, ())
        self.k_cre = tf.placeholder_with_default(1e-3, ())
        self.ϵ = tf.placeholder_with_default(0.01, ())
        n_pts = tf.shape(self.x0)[0]
        route_cr(self.root, tf.ones((n_pts,)), tf.ones((n_pts,)),
                 Namespace(arch=arch, k_l2=k_l2, k_cpt=self.k_cpt,
                           k_cre=self.k_cre, ϵ=self.ϵ, mode=self.mode))
        c_err = sum(ℓ.p_tr * ℓ.c_err for ℓ in self.layers)
        c_cpt = sum(ℓ.p_tr * self.k_cpt * ℓ.n_ops for ℓ in self.layers)
        c_cre = sum(ℓ.p_tr * ℓ.c_cre for ℓ in self.layers)
        c_mod = sum(ℓ.p_tr * (ℓ.c_mod + ℓ.router.c_mod) for ℓ in self.layers)
        c_tr = c_err + c_cpt + c_cre + c_mod
        self._train_op = minimize_expected(
            self, tf.reduce_mean(c_tr), optimizer)

    def train(self, x0, y, hypers):
        self._train_op.run({self.x0: x0, self.y: y, self.mode: 'tr', **hypers})
