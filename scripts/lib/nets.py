from abc import ABCMeta, abstractmethod
from functools import reduce
from types import SimpleNamespace as Namespace

import numpy as np
import tensorflow as tf

from lib.layers import BatchNorm, Chain, LinTrans, Rect

################################################################################
# Routing Layers
################################################################################

def router(n_act, arch, k_l2):
    return Chain({},
        sum(([LinTrans(dict(n_chan=n, k_l2=k_l2)), BatchNorm({}), Rect({})]
             for n in arch), []) + [LinTrans(dict(n_chan=n_act, k_l2=k_l2))])

################################################################################
# Optimization
################################################################################

def minimize_expected(net, cost, optimizer, lr_routing_scale=1.0):
    lr_scales = {
        **{θ: 1 / tf.sqrt(tf.reduce_mean(tf.square(ℓ.p_tr)))
           for ℓ in net.layers for θ in vars(ℓ.params).values()},
        **{θ: 1 / tf.sqrt(tf.reduce_mean(tf.square(ℓ.p_tr))) * lr_routing_scale
           for ℓ in net.layers for θ in vars(ℓ.router.params).values()}}
    grads = optimizer.compute_gradients(cost)
    scaled_grads = [(lr_scales[θ] * g, θ) for g, θ in grads if g is not None]
    return optimizer.apply_gradients(scaled_grads)

################################################################################
# Root Network Class
################################################################################

class Net(metaclass=ABCMeta):
    default_hypers = {}

    def __init__(self, x0_shape, y_shape, hypers, root):
        full_hyper_dict = {**self.__class__.default_hypers, **hypers}
        self.hypers = Namespace(**full_hyper_dict)
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

    def validate(self, x0, y, hypers):
        pass

    def eval(self, target, x0, y, hypers):
        pass

################################################################################
# Statically-Routed Networks
################################################################################

class SRNet(Net):
    def __init__(self, x0_shape, y_shape, optimizer, root):
        super().__init__(x0_shape, y_shape, {}, root)
        for ℓ in self.layers:
            ℓ.p_ev = tf.ones((tf.shape(ℓ.x)[0],))
        c_tr = sum(ℓ.c_err + ℓ.c_mod for ℓ in self.layers)
        self._train_op = optimizer.minimize(tf.reduce_mean(c_tr))
        self._sess = tf.Session()
        self._sess.run(tf.initialize_all_variables())

    def __del__(self):
        self._sess.close()

    def train(self, x0, y, hypers):
        self._sess.run(self._train_op, {
            self.x0: x0, self.y: y, self.mode: 'tr', **hypers})

    def eval(self, target, x0, y, hypers):
        return self._sess.run(target, {
            self.x0: x0, self.y: y, **hypers})

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
    ℓ.router = router(len(ℓ.sinks), opts.arch, opts.k_l2)
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
    ℓ.μ_tr = tf.Variable(0.0, trainable=False)
    ℓ.μ_vl = tf.Variable(0.0, trainable=False)
    μ_batch = tf.reduce_sum(ℓ.p_tr * ℓ.c_err) / tf.reduce_sum(ℓ.p_tr)
    ℓ.update_μ_tr = tf.assign(ℓ.μ_tr, opts.λ * ℓ.μ_tr + (1 - opts.λ) * μ_batch)
    ℓ.update_μ_vl = tf.assign(ℓ.μ_vl, opts.λ * ℓ.μ_vl + (1 - opts.λ) * μ_batch)
    ℓ.c_gen = ℓ.μ_vl - ℓ.μ_tr

class DSNet(Net):
    default_hypers = dict(arch=[], k_cpt=0.0, k_l2=0.0, ϵ=0.1, λ=0.99)

    def __init__(self, x0_shape, y_shape, optimizer, hypers, root):
        super().__init__(x0_shape, y_shape, hypers, root)
        n_pts = tf.shape(self.x0)[0]
        route_ds(self.root, tf.ones((n_pts,)), tf.ones((n_pts,)),
                 Namespace(mode=self.mode, **vars(self.hypers)))
        c_err = sum(ℓ.p_tr * ℓ.c_err for ℓ in self.layers)
        c_gen = sum(ℓ.p_tr * ℓ.c_gen for ℓ in self.layers)
        c_cpt = sum(ℓ.p_tr * self.hypers.k_cpt * ℓ.n_ops for ℓ in self.layers)
        c_mod = sum(tf.stop_gradient(ℓ.p_tr) * (ℓ.c_mod + ℓ.router.c_mod)
                    for ℓ in self.layers)
        c_tr = c_err + c_gen + c_cpt + c_mod
        with tf.control_dependencies([ℓ.update_μ_tr for ℓ in self.layers]):
            self._train_op = minimize_expected(
                self, tf.reduce_mean(c_tr), optimizer)
        self._validate_op = tf.group(*(ℓ.update_μ_vl for ℓ in self.layers))
        self._sess = tf.Session()
        self._sess.run(tf.initialize_all_variables())

    def __del__(self):
        self._sess.close()

    def train(self, x0, y, hypers):
        self._sess.run(self._train_op, {
            self.x0: x0, self.y: y, self.mode: 'tr', **hypers})

    def validate(self, x0, y, hypers):
        self._sess.run(self._validate_op, {
            self.x0: x0, self.y: y, **hypers})

    def eval(self, target, x0, y, hypers):
        return self._sess.run(target, {
            self.x0: x0, self.y: y, **hypers})

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
    ℓ.c_ev = (
        ℓ.c_err + opts.k_cpt * ℓ.n_ops
        + sum(s.c_ev for s in ℓ.sinks))
    ℓ.c_opt = (
        ℓ.c_err + opts.k_cpt * ℓ.n_ops
        + sum(s.c_opt for s in ℓ.sinks))
    ℓ.c_cre = 0.0

def route_cr_dyn(ℓ, p_tr, p_ev, opts):
    ℓ.p_tr = p_tr
    ℓ.p_ev = p_ev
    ℓ.router = router(len(ℓ.sinks), opts.arch, opts.k_l2)
    ℓ.router.link(Namespace(x=ℓ.x, mode=opts.mode))
    π_ev = tf.to_float(tf.equal(
        tf.expand_dims(tf.to_int32(tf.argmin(ℓ.router.x, 1)), 1),
        tf.range(len(ℓ.sinks))))
    π_tr = opts.ϵ / len(ℓ.sinks) + (1 - opts.ϵ) * π_ev
    for i, s in enumerate(ℓ.sinks):
        route_cr(s, ℓ.p_tr * π_tr[:, i], ℓ.p_ev * π_ev[:, i], opts)
    ℓ.c_ev = (
        ℓ.c_err + ℓ.c_gen + opts.k_cpt * ℓ.n_ops
        + sum(π_ev[:, i] * s.c_ev
              for i, s in enumerate(ℓ.sinks)))
    ℓ.c_opt = (
        ℓ.c_err + ℓ.c_gen + opts.k_cpt * ℓ.n_ops
        + reduce(tf.minimum, (s.c_opt for s in ℓ.sinks)))
    if opts.optimistic:
        ℓ.c_cre = opts.k_cre * sum(
            π_tr[:, i] * tf.square(
                ℓ.router.x[:, i] - tf.stop_gradient(s.c_opt))
            for i, s in enumerate(ℓ.sinks))
    else:
        ℓ.c_cre = opts.k_cre * sum(
            π_tr[:, i] * tf.square(
                ℓ.router.x[:, i] - tf.stop_gradient(s.c_ev))
            for i, s in enumerate(ℓ.sinks))

def route_cr(ℓ, p_tr, p_ev, opts):
    ℓ.μ_tr = tf.Variable(0.0, trainable=False)
    ℓ.μ_vl = tf.Variable(0.0, trainable=False)
    μ_batch = tf.reduce_sum(ℓ.p_tr * ℓ.c_err) / tf.reduce_sum(ℓ.p_tr)
    ℓ.update_μ_tr = tf.assign(ℓ.μ_tr, opts.λ * ℓ.μ_tr + (1 - opts.λ) * μ_batch)
    ℓ.update_μ_vl = tf.assign(ℓ.μ_vl, opts.λ * ℓ.μ_vl + (1 - opts.λ) * μ_batch)
    ℓ.c_gen = ℓ.μ_vl - ℓ.μ_tr
    if len(ℓ.sinks) < 2: route_cr_stat(ℓ, p_tr, p_ev, opts)
    else: route_cr_dyn(ℓ, p_tr, p_ev, opts)

class CRNet(Net):
    default_hypers = dict(
        arch=[], k_cpt=0.0, k_cre=1e-3, k_l2=0.0, ϵ=0.1,
        optimistic=False)

    def __init__(self, x0_shape, y_shape, optimizer, hypers, root):
        super().__init__(x0_shape, y_shape, hypers, root)
        n_pts = tf.shape(self.x0)[0]
        route_cr(self.root, tf.ones((n_pts,)), tf.ones((n_pts,)),
                 Namespace(mode=self.mode, **vars(self.hypers)))
        c_err = sum(ℓ.p_tr * ℓ.c_err for ℓ in self.layers)
        c_cpt = sum(ℓ.p_tr * self.hypers.k_cpt * ℓ.n_ops for ℓ in self.layers)
        c_cre = sum(ℓ.p_tr * ℓ.c_cre for ℓ in self.layers)
        c_mod = sum(ℓ.p_tr * (ℓ.c_mod + ℓ.router.c_mod) for ℓ in self.layers)
        c_tr = c_err + c_cpt + c_cre + c_mod
        with tf.control_dependencies([ℓ.update_μ_tr for ℓ in self.layers]):
            self._train_op = minimize_expected(
                self, tf.reduce_mean(c_tr), optimizer)
        self._validate_op = tf.group(*(ℓ.update_μ_vl for ℓ in self.layers))
        self._sess = tf.Session()
        self._sess.run(tf.initialize_all_variables())

    def __del__(self):
        self._sess.close()

    def train(self, x0, y, hypers):
        self._sess.run(self._train_op, {
            self.x0: x0, self.y: y, self.mode: 'tr', **hypers})

    def validate(self, x0, y, hypers):
        self._sess.run(self._validate_op, {
            self.x0: x0, self.y: y, **hypers})

    def eval(self, target, x0, y, hypers):
        return self._sess.run(target, {
            self.x0: x0, self.y: y, **hypers})
