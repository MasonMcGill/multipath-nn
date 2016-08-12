from abc import ABCMeta, abstractmethod
from functools import reduce
from types import SimpleNamespace as Namespace

import numpy as np
import tensorflow as tf

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

class DSNet(Net):
    def __init__(self, x0_shape, y_shape, optimizer, root):
        super().__init__(x0_shape, y_shape, root)
        self.k_cpt = tf.placeholder_with_default(0.0, ())
        self.k_l2 = tf.placeholder_with_default(0.0, ())
        self.ϵ = tf.placeholder_with_default(0.01, ())
        def route_stat(ℓ, p_tr, p_ev):
            ℓ.p_tr = p_tr
            ℓ.p_ev = p_ev
            for s in ℓ.sinks:
                route(s, ℓ.p_tr, ℓ.p_ev)
        def route_dyn(ℓ, p_tr, p_ev):
            ℓ.p_tr = p_tr
            ℓ.p_ev = p_ev
            n_chan_in = np.prod(ℓ.x.get_shape().as_list()[1:])
            w_scale = 1 / np.sqrt(n_chan_in)
            w_shape = (n_chan_in, len(ℓ.sinks))
            ℓ.router.w = tf.Variable(w_scale * tf.random_normal(w_shape))
            ℓ.router.b = tf.Variable(tf.zeros(len(ℓ.sinks)))
            ℓ.router.c_mod = self.k_l2 * tf.reduce_sum(tf.square(ℓ.router.w))
            x_flat = tf.reshape(ℓ.x, (-1, n_chan_in))
            s = tf.matmul(x_flat, ℓ.router.w) + ℓ.router.b
            π_tr = self.ϵ / len(ℓ.sinks) + (1 - self.ϵ) * tf.nn.softmax(s)
            π_ev = tf.to_float(tf.equal(
                tf.expand_dims(tf.to_int32(tf.argmax(s, 1)), 1),
                tf.range(len(ℓ.sinks))))
            for i, s in enumerate(ℓ.sinks):
                route(s, ℓ.p_tr * π_tr[:, i], ℓ.p_ev * π_ev[:, i])
        def route(ℓ, p_tr, p_ev):
            if len(ℓ.sinks) < 2: route_stat(ℓ, p_tr, p_ev)
            else: route_dyn(ℓ, p_tr, p_ev)
        n_pts = tf.shape(self.x0)[0]
        route(self.root, tf.ones((n_pts,)), tf.ones((n_pts,)))
        c_err = sum(ℓ.p_tr * ℓ.c_err for ℓ in self.layers)
        c_cpt = sum(ℓ.p_tr * self.k_cpt * ℓ.n_ops for ℓ in self.layers)
        c_mod = sum(
            tf.stop_gradient(ℓ.p_tr)
            * (ℓ.c_mod + getattr(ℓ.router, 'c_mod', 0.0))
            for ℓ in self.layers)
        c_tr = c_err + c_cpt + c_mod
        lr_scales = {
            p: 1 / tf.sqrt(tf.reduce_mean(tf.square(ℓ.p_tr)))
            for ℓ in self.layers for p in [
                *vars(ℓ.params).values(),
                *vars(ℓ.router).values()]}
        grads = optimizer.compute_gradients(tf.reduce_mean(c_tr))
        scaled_grads = [(lr_scales[p] * g, p) for g, p in grads]
        self._train_op = optimizer.apply_gradients(scaled_grads)

    def train(self, x0, y, hypers):
        self._train_op.run({self.x0: x0, self.y: y, self.mode: 'tr', **hypers})

################################################################################
# Cost Regression Networks
################################################################################

class CRNet(Net):
    def __init__(self, x0_shape, y_shape, optimizer, root):
        super().__init__(x0_shape, y_shape, root)
        self.k_cpt = tf.placeholder_with_default(0.0, ())
        self.k_l2 = tf.placeholder_with_default(0.0, ())
        self.k_cre = tf.placeholder_with_default(1e-3, ())
        self.ϵ = tf.placeholder_with_default(0.01, ())
        def route_stat(ℓ, p_tr, p_ev):
            ℓ.p_tr = p_tr
            ℓ.p_ev = p_ev
            for s in ℓ.sinks:
                route(s, ℓ.p_tr, ℓ.p_ev)
            ℓ.c_cre = 0.0
            ℓ.c_ev = (
                ℓ.c_err + self.k_cpt * ℓ.n_ops
                + sum(s.c_ev for s in ℓ.sinks))
        def route_dyn(ℓ, p_tr, p_ev):
            ℓ.p_tr = p_tr
            ℓ.p_ev = p_ev
            n_chan_in = np.prod(ℓ.x.get_shape().as_list()[1:])
            w_scale = 1 / np.sqrt(n_chan_in)
            w_shape = (n_chan_in, len(ℓ.sinks))
            ℓ.router.w = tf.Variable(w_scale * tf.random_normal(w_shape))
            ℓ.router.b = tf.Variable(tf.zeros(len(ℓ.sinks)))
            ℓ.router.c_mod = self.k_l2 * tf.reduce_sum(tf.square(ℓ.router.w))
            x_flat = tf.reshape(ℓ.x, (-1, n_chan_in))
            c_est = tf.matmul(x_flat, ℓ.router.w) + ℓ.router.b
            π_ev = tf.to_float(tf.equal(
                tf.expand_dims(tf.to_int32(tf.argmin(c_est, 1)), 1),
                tf.range(len(ℓ.sinks))))
            π_tr = self.ϵ / len(ℓ.sinks) + (1 - self.ϵ) * π_ev
            for i, s in enumerate(ℓ.sinks):
                route(s, ℓ.p_tr * π_tr[:, i], ℓ.p_ev * π_ev[:, i])
            ℓ.c_cre = self.k_cre * sum(
                π_tr[:, i] * tf.square(tf.stop_gradient(s.c_ev) - c_est[:, i])
                for i, s in enumerate(ℓ.sinks))
            ℓ.c_ev = (
                ℓ.c_err + self.k_cpt * ℓ.n_ops
                + sum(π_ev[:, i] * s.c_ev
                      for i, s in enumerate(ℓ.sinks)))
        def route(ℓ, p_tr, p_ev):
            if len(ℓ.sinks) < 2: route_stat(ℓ, p_tr, p_ev)
            else: route_dyn(ℓ, p_tr, p_ev)
        n_pts = tf.shape(self.x0)[0]
        route(self.root, tf.ones((n_pts,)), tf.ones((n_pts,)))
        c_err = sum(ℓ.p_tr * ℓ.c_err for ℓ in self.layers)
        c_cpt = sum(ℓ.p_tr * self.k_cpt * ℓ.n_ops for ℓ in self.layers)
        c_cre = sum(ℓ.p_tr * ℓ.c_cre for ℓ in self.layers)
        c_mod = sum(
            tf.stop_gradient(ℓ.p_tr)
            * (ℓ.c_mod + getattr(ℓ.router, 'c_mod', 0.0))
            for ℓ in self.layers)
        c_tr = c_err + c_cpt + c_cre + c_mod
        lr_scales = {
            p: 1 / tf.sqrt(tf.reduce_mean(tf.square(ℓ.p_tr)))
            for ℓ in self.layers for p in [
                *vars(ℓ.params).values(),
                *vars(ℓ.router).values()]}
        grads = optimizer.compute_gradients(tf.reduce_mean(c_tr))
        scaled_grads = [(lr_scales.get(p, 1.0) * g, p) for g, p in grads]
        self._train_op = optimizer.apply_gradients(scaled_grads)

    def train(self, x0, y, hypers):
        self._train_op.run({self.x0: x0, self.y: y, self.mode: 'tr', **hypers})
