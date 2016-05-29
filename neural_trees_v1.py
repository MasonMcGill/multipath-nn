import itertools
import os

os.environ['THEANO_FLAGS'] = (
    'device=gpu,floatX=float32,cast_policy=numpy+floatX,' +
    'enable_initial_driver_test=False,warn_float64=raise')

import numpy as np
import numpy.random as rand
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
        self.n_in = n_in
        self.n_out = n_out
        self.n_ops = self.w.size.eval()

    def clone(self):
        res = ReLuTF.__new__(ReLuTF)
        res.w = th.shared(self.w.get_value())
        res.b = th.shared(self.b.get_value())
        res.n_in = self.n_in
        res.n_out = self.n_out
        res.n_ops = self.n_ops
        return res

    def params(self):
        return [self.w, self.b]

    def link(self, x, k_cpt, k_l2):
        self.x = nn.relu(ts.dot(x, self.w) + self.b)
        self.c_cpt = k_cpt * ts.cast(self.n_ops, 'float32')
        self.l_l2 = k_l2 * ts.sum(ts.sqr(self.w))

class IdentityTF:
    def __init__(self, n_in=784):
        self.n_ops = 0
        self.n_in = n_in
        self.n_out = n_in

    def clone(self):
        res = IdentityTF.__new__(IdentityTF)
        res.n_ops = self.n_ops
        res.n_in = self.n_in
        res.n_out = self.n_out
        return res

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
        self.n_in = n_in
        self.w_scale = w_scale
        self.w = th.shared(np.float32(w_scale * rand.randn(n_in, n_act)))
        self.b = th.shared(np.float32(w_scale * rand.randn(n_act)))
        self.tf = tf
        self.children = children

    def clone(self):
        res = Layer.__new__(Layer)
        res.n_in = self.n_in
        res.w_scale = self.w_scale
        res.w = th.shared(self.w.get_value())
        res.b = th.shared(self.b.get_value())
        res.tf = self.tf.clone()
        res.children = [ch.clone() for ch in self.children]
        return res

    def clone_replacing(self, i, child):
        res = Layer.__new__(Layer)
        res.n_in = self.n_in
        res.w_scale = self.w_scale
        res.w = th.shared(self.w.get_value())
        res.b = th.shared(self.b.get_value())
        res.tf = self.tf.clone()
        res.children = [child if j == i else ch.clone()
                        for j, ch in enumerate(self.children)]
        return res

    def clone_adding(self, child):
        res = Layer.__new__(Layer)
        res.n_in = self.n_in
        res.w_scale = self.w_scale
        res.w = th.shared(np.hstack([
            self.w.get_value(),
            np.float32(self.w_scale * rand.randn(self.n_in, 1))]))
        res.b = th.shared(np.hstack([
            self.b.get_value(),
            np.float32(self.w_scale * rand.randn(1))]))
        res.tf = self.tf.clone()
        res.children = [ch.clone() for ch in self.children] + [child]
        return res

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
        c_cpt = self.tf.c_cpt
        c_err = ts.cast(ts.neq(ts.arange(n_lab), y), 'float32')
        c_del = [ch.c_fav_act for ch in self.children]
        c_act = c_cpt + ts.concatenate([c_err] + c_del, axis=1)
        self.c_fav_act = ts.sum(δ_fav_act * c_act, axis=1, keepdims=True)

        # compute the local loss
        l_err = ts.sum(ts.sqr(c_act_est - c_act), axis=1, keepdims=True)
        l_l2 = k_l2 * ts.sum(ts.sqr(self.w)) + self.tf.l_l2
        self.l_lay = p * (l_err + l_l2)

        # compute global loss
        self.mean_path_len = p + sum(ch.mean_path_len for ch in self.children)
        self.l_tot = self.l_lay + sum(ch.l_tot for ch in self.children)
        self.l_tot_mean = self.l_tot / self.mean_path_len

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
        c_avg = ts.mean(root.c_fav_act)
        l_avg = ts.mean(root.l_tot_mean)
        self.train = th.function([x, y, k_cpt, k_l2, ϵ, λ], [l_avg, c_avg], updates=[
            (p, p - λ * ts.grad(l_avg, p))
            for p in root.params()
        ], on_unused_input='ignore')
        self.classify = th.function([x], root.y_est)
        self.root = root
