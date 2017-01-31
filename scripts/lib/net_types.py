from abc import ABCMeta
from functools import reduce
from types import SimpleNamespace as Ns

import numpy as np
import tensorflow as tf

from lib.layer_types import BatchNorm, Chain, Layer, LinTrans, NoOp, Rect

################################################################################
# Support Functions
################################################################################

def n_leaves(ℓ): return (
    1 if len(ℓ.sinks) == 0
    else sum(map(n_leaves, ℓ.sinks)))

def params_list_rec(ℓ):
    if ℓ is not None:
        yield from vars(ℓ.params).values()
        for c in getattr(ℓ, 'comps', []):
            yield from params_list_rec(c)

def minimize_expectation(layers, cost, optimizer, α_rtr=1, talr=True):
    lr_scale = lambda ℓ: (
        1 / tf.sqrt(tf.reduce_mean(tf.square(ℓ.p_tr)))
        if talr else 1)
    lr_scales = {
        **{θ: lr_scale(ℓ)
           for ℓ in layers
           for θ in params_list_rec(ℓ)},
        **{θ: α_rtr * lr_scale(ℓ)
           for ℓ in layers
           for θ in params_list_rec(ℓ.router)}}
    grads = optimizer.compute_gradients(cost)
    scaled_grads = [(lr_scales[θ] * g, θ) for g, θ in grads if g is not None]
    return optimizer.apply_gradients(scaled_grads)

################################################################################
# Root Network Class
################################################################################

class Net(metaclass=ABCMeta):
    default_hypers = Ns(x0_shape=(), y_shape=())

    def __init__(self, **options):
        self.root = options.pop('root', NoOp())
        self.hypers = Ns(**{**vars(type(self).default_hypers), **options})
        self.params = Ns()
        self.x0 = tf.placeholder(tf.float32, (None,) + self.hypers.x0_shape)
        self.y = tf.placeholder(tf.float32, (None,) + self.hypers.y_shape)
        self.mode = tf.placeholder_with_default('ev', ())
        self.train = tf.no_op()
        self.link()

    def link(self):
        def link_layer(ℓ, x, y, mode):
            ℓ.link(x, y, mode)
            if ℓ.router is not None:
                ℓ.router.link(ℓ.x, y, mode)
            for s in ℓ.sinks:
                link_layer(s, ℓ.x, y, mode)
        link_layer(self.root, self.x0, self.y, self.mode)

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

    @property
    def switches(self):
        return (ℓ for ℓ in self.layers if len(ℓ.sinks) > 1)

################################################################################
# Statically-Routed Networks
################################################################################

class SRNet(Net):
    default_hypers = Ns(λ_lrn=1e-3, μ_lrn=0.9)

    def link(self):
        super().link()
        ϕ = self.hypers
        self.λ_lrn = tf.placeholder_with_default(ϕ.λ_lrn, ())
        self.μ_lrn = tf.placeholder_with_default(ϕ.μ_lrn, ())
        for ℓ in self.layers:
            ℓ.p_ev = tf.ones((tf.shape(self.x0)[0],))
        c_tot = tf.reduce_mean(sum(ℓ.c_err + ℓ.c_mod for ℓ in self.layers))
        opt = tf.train.MomentumOptimizer(self.λ_lrn, self.μ_lrn)
        self.train = opt.minimize(c_tot)

################################################################################
# Decision Smoothing Networks
################################################################################

class DSNet(Net):
    default_hypers = Ns(
        k_cpt=0.0, k_dec=0.01, ϵ=1e-6, τ=1.0, λ_lrn=1e-3, μ_lrn=0.9,
        dyn_k_cpt=False, talr=True, α_rtr=1.0)

    def _route(self, ℓ, p_tr, p_ev):
        ℓ.p_tr = p_tr
        ℓ.p_ev = p_ev
        if len(ℓ.sinks) < 2:
            self._route_sinks_stat(ℓ)
        else:
            self._route_sinks_dyn(ℓ)

    def _route_sinks_stat(self, ℓ):
        for s in ℓ.sinks:
            self._route(s, ℓ.p_tr, ℓ.p_ev)

    def _route_sinks_dyn(self, ℓ):
        def p_tr_ϵ(ℓ):
            return self.ϵ * n_leaves(ℓ) / n_leaves(self.root)
        π_tr = (
            (1 - p_tr_ϵ(ℓ) / ℓ.p_tr[:, None])
            * tf.nn.softmax(ℓ.router.x / self.τ)
            + list(map(p_tr_ϵ, ℓ.sinks)) / ℓ.p_tr[:, None])
        π_ev = tf.to_float(tf.equal(
            tf.expand_dims(tf.to_int32(tf.argmax(ℓ.router.x, 1)), 1),
            tf.range(len(ℓ.sinks))))
        for i, s in enumerate(ℓ.sinks):
            self._route(s, ℓ.p_tr * π_tr[:, i], ℓ.p_ev * π_ev[:, i])

    def _c_dec(self, ℓ):
        dims = tuple(range(1, len(ℓ.router.x.get_shape())))
        return self.hypers.k_dec * tf.reduce_sum(tf.square(ℓ.router.x), dims)

    def link(self):
        ϕ = self.hypers
        self.λ_lrn = tf.placeholder_with_default(ϕ.λ_lrn, ())
        self.μ_lrn = tf.placeholder_with_default(ϕ.μ_lrn, ())
        self.ϵ = tf.placeholder_with_default(ϕ.ϵ, ())
        self.τ = tf.placeholder_with_default(ϕ.τ, ())
        self.k_cpt = (
            tf.placeholder(tf.float32, (None,))
            if ϕ.dyn_k_cpt else ϕ.k_cpt)
        def link_layer(ℓ, x, y, mode):
            ℓ.link(x, y, mode)
            if ℓ.router is not None:
                concat_k_cpt = lambda x_: tf.concat(1, [
                    tf.reshape(x_, (
                        tf.shape(x_)[0],
                        np.prod(x_.get_shape().as_list()[1:]))),
                    self.k_cpt[:, None]
                    * tf.ones((tf.shape(x_)[0], 1))])
                if not ϕ.dyn_k_cpt:
                    x_rte = ℓ.x
                elif isinstance(ℓ.x, list):
                    x_rte = list(map(concat_k_cpt, ℓ.x))
                else:
                    x_rte = concat_k_cpt(ℓ.x)
                ℓ.router.link(x_rte, y, mode)
            for s in ℓ.sinks:
                link_layer(s, ℓ.x, y, mode)
        link_layer(self.root, self.x0, self.y, self.mode)
        n_pts = tf.shape(self.x0)[0]
        self._route(self.root, tf.ones((n_pts,)), tf.ones((n_pts,)))
        c_err = sum(ℓ.p_tr * ℓ.c_err for ℓ in self.layers)
        c_cpt = sum(
            ℓ.p_tr * self.k_cpt * (ℓ.n_ops + getattr(ℓ.router, 'n_ops', 0))
            for ℓ in self.layers)
        c_mod = sum(
            tf.stop_gradient(ℓ.p_tr) * (ℓ.c_mod + getattr(ℓ.router, 'c_mod', 0))
            for ℓ in self.layers)
        c_dec = sum(
            tf.stop_gradient(ℓ.p_tr) * self._c_dec(ℓ)
            for ℓ in self.switches)
        c_tot = tf.reduce_mean(c_err + c_cpt + c_mod + c_dec)
        opt = tf.train.MomentumOptimizer(self.λ_lrn, self.μ_lrn)
        self.train = minimize_expectation(
            list(self.layers), c_tot, opt,
            self.hypers.α_rtr, self.hypers.talr)

################################################################################
# Cost Regression Networks
################################################################################

class CRNet(Net):
    default_hypers = Ns(
        k_cpt=0.0, k_cre=1e-3, ϵ=1e-6, τ=0.01, optimistic=False,
        use_cls_err=False, λ_lrn=1e-3, μ_lrn=0.9, talr=True, α_rtr=1.0)

    def _route(self, ℓ, p_tr, p_ev):
        ℓ.p_tr = p_tr
        ℓ.p_ev = p_ev
        if len(ℓ.sinks) < 2:
            self._route_sinks_stat(ℓ)
        else:
            self._route_sinks_dyn(ℓ)

    def _route_sinks_stat(self, ℓ):
        for s in ℓ.sinks:
            self._route(s, ℓ.p_tr, ℓ.p_ev)
        ϕ = self.hypers
        c_err = (
            (1 - getattr(ℓ, 'δ_cor', 1))
            if ϕ.use_cls_err else ℓ.c_err)
        ℓ.c_ev = (
            c_err + ϕ.k_cpt * ℓ.n_ops
            + sum(s.c_ev for s in ℓ.sinks))
        ℓ.c_opt = (
            c_err + ϕ.k_cpt * ℓ.n_ops
            + sum(s.c_opt for s in ℓ.sinks))
        ℓ.c_cre = 0

    def _route_sinks_dyn(self, ℓ):
        def p_tr_ϵ(ℓ):
            return self.ϵ * n_leaves(ℓ) / n_leaves(self.root)
        ϕ = self.hypers
        c_err = (
            (1 - getattr(ℓ, 'δ_cor', 1))
            if ϕ.use_cls_err else ℓ.c_err)
        π_tr = (
            (1 - p_tr_ϵ(ℓ) / ℓ.p_tr[:, None])
            * tf.nn.softmax(ℓ.router.x / self.τ)
            + list(map(p_tr_ϵ, ℓ.sinks)) / ℓ.p_tr[:, None])
        π_ev = tf.to_float(tf.equal(
            tf.expand_dims(tf.to_int32(tf.argmax(ℓ.router.x, 1)), 1),
            tf.range(len(ℓ.sinks))))
        for i, s in enumerate(ℓ.sinks):
            self._route(s, ℓ.p_tr * π_tr[:, i], ℓ.p_ev * π_ev[:, i])
        ℓ.c_ev = (
            c_err + ϕ.k_cpt * (ℓ.n_ops + ℓ.router.n_ops)
            + sum(π_ev[:, i] * s.c_ev
                  for i, s in enumerate(ℓ.sinks)))
        ℓ.c_opt = (
            c_err + ϕ.k_cpt * (ℓ.n_ops + ℓ.router.n_ops)
            + reduce(tf.minimum, (s.c_opt for s in ℓ.sinks)))
        ℓ.c_cre = (
            ϕ.k_cre * sum(
                tf.square(ℓ.router.x[:, i] + tf.stop_gradient(
                    s.c_opt if ϕ.optimistic else s.c_ev))
                for i, s in enumerate(ℓ.sinks)))

    def link(self):
        super().link()
        ϕ = self.hypers
        self.λ_lrn = tf.placeholder_with_default(ϕ.λ_lrn, ())
        self.μ_lrn = tf.placeholder_with_default(ϕ.μ_lrn, ())
        self.ϵ = tf.placeholder_with_default(ϕ.ϵ, ())
        self.τ = tf.placeholder_with_default(ϕ.τ, ())
        n_pts = tf.shape(self.x0)[0]
        self._route(self.root, tf.ones((n_pts,)), tf.ones((n_pts,)))
        c_err = sum(tf.stop_gradient(ℓ.p_tr) * ℓ.c_err for ℓ in self.layers)
        c_cre = sum(tf.stop_gradient(ℓ.p_tr) * ℓ.c_cre for ℓ in self.layers)
        c_mod = sum(
            tf.stop_gradient(ℓ.p_tr) * (ℓ.c_mod + getattr(ℓ.router, 'c_mod', 0))
            for ℓ in self.layers)
        c_tot = tf.reduce_mean(c_err + c_cre + c_mod)
        opt = tf.train.MomentumOptimizer(self.λ_lrn, self.μ_lrn)
        self.train = minimize_expectation(
            list(self.layers), c_tot, opt,
            self.hypers.α_rtr, self.hypers.talr)
