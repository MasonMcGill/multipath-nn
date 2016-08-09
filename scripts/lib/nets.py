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

################################################################################
# Statically-Routed Networks
################################################################################

class SRNet(Net):
    def __init__(self, x0_shape, y_shape, root):
        super().__init__(x0_shape, y_shape, root)
        for ℓ in self.layers:
            ℓ.p_tr = tf.ones((tf.shape(ℓ.x)[0],))
            ℓ.p_ev = tf.ones((tf.shape(ℓ.x)[0],))
        self.c_tr = sum(ℓ.c_err + ℓ.c_mod for ℓ in self.layers)

################################################################################
# Decision Smoothing Networks
################################################################################

class DSNet(Net):
    def __init__(self, x0_shape, y_shape, root):
        super().__init__(x0_shape, y_shape, root)
        self.k_cpt = tf.placeholder_with_default(0.0, ())
        self.k_l2 = tf.placeholder_with_default(0.0, ())
        self.ϵ = tf.placeholder_with_default(0.01, ())
        c_mod = 0.0
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
            w = tf.Variable(w_scale * tf.random_normal(w_shape))
            b = tf.Variable(tf.zeros(len(ℓ.sinks)))
            x_flat = tf.reshape(ℓ.x, (-1, n_chan_in))
            s = tf.matmul(x_flat, w) + b
            π_tr = (
                self.ϵ / len(ℓ.sinks)
                + (1 - self.ϵ) * tf.nn.softmax(s))
            π_ev = tf.to_float(tf.equal(
                tf.expand_dims(tf.to_int32(tf.argmax(s, 1)), 1),
                tf.range(len(ℓ.sinks))))
            for i, s in enumerate(ℓ.sinks):
                route(s, ℓ.p_tr * π_tr[:, i], ℓ.p_ev * π_ev[:, i])
            nonlocal c_mod
            c_mod += (
                tf.stop_gradient(ℓ.p_tr)
                * self.k_l2 * tf.reduce_sum(tf.square(w)))
        def route(ℓ, p_tr, p_ev):
            if len(ℓ.sinks) < 2: route_stat(ℓ, p_tr, p_ev)
            else: route_dyn(ℓ, p_tr, p_ev)
        route(self.root, 1.0, 1.0)
        c_err = sum(ℓ.p_tr * ℓ.c_err for ℓ in self.layers)
        c_cpt = sum(ℓ.p_tr * self.k_cpt * ℓ.n_ops for ℓ in self.layers)
        c_mod += sum(tf.stop_gradient(ℓ.p_tr) * ℓ.c_mod for ℓ in self.layers)
        self.c_tr = c_err + c_cpt + c_mod
        # To-do: remove exploitation bias

################################################################################
# Cost Regression Networks
################################################################################

class CRNet(Net):
    def __init__(self, x0_shape, y_shape, root):
        super().__init__(x0_shape, y_shape, root)
        self.k_cpt = tf.placeholder_with_default(0.0, ())
        self.k_l2 = tf.placeholder_with_default(0.0, ())
        self.k_cre = tf.placeholder_with_default(0.01, ())
        self.ϵ = tf.placeholder_with_default(0.1, ())
        c_mod = 0.0
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
            w = tf.Variable(w_scale * tf.random_normal(w_shape))
            b = tf.Variable(tf.zeros(len(ℓ.sinks)))
            x_flat = tf.reshape(ℓ.x, (-1, n_chan_in))
            c_est = tf.matmul(x_flat, w) + b
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
            nonlocal c_mod
            c_mod += (
                tf.stop_gradient(ℓ.p_tr)
                * self.k_l2 * tf.reduce_sum(tf.square(w)))
        def route(ℓ, p_tr, p_ev):
            if len(ℓ.sinks) < 2: route_stat(ℓ, p_tr, p_ev)
            else: route_dyn(ℓ, p_tr, p_ev)
        route(self.root, 1.0, 1.0)
        c_err = sum(ℓ.p_tr * ℓ.c_err for ℓ in self.layers)
        c_cpt = sum(ℓ.p_tr * self.k_cpt * ℓ.n_ops for ℓ in self.layers)
        c_cre = sum(ℓ.p_tr * ℓ.c_cre for ℓ in self.layers)
        c_mod += sum(ℓ.p_tr * ℓ.c_mod for ℓ in self.layers)
        self.c_tr = c_err + c_cpt + c_cre + c_mod
