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

    def params(self):
        return list(itertools.chain(
            [self.w, self.b], self.tf.params(),
            *[c.params() for c in self.children]))

    def link(self, x, y, k_cpt, k_l2, ϵ):
        # infer activity shape
        n_lab = self.w.shape[1].eval() - len(self.children)
        n_act = self.w.shape[1].eval()

        # link to the transformation
        self.tf.link(x, k_cpt, k_l2)

        # propagate activity
        c_act_est = ts.dot(self.tf.x, self.w) + self.b
        i_fav_act = ts.argmin(c_act_est, axis=1, keepdims=True)
        π_ts = ts.eq(ts.arange(n_act), i_fav_act)
        # π_tr = ϵ / np.float32(n_act - 1) + (1 - ϵ - ϵ / np.float32(n_act - 1)) * π_ts
        π_tr = nn.softmax(-π_ts / 4)

        # link recursively
        for i, ch in enumerate(self.children):
            ch.link(self.tf.x, y, k_cpt, k_l2, ϵ)

        # perform error analysis
        c_err = ts.cast(ts.neq(ts.arange(n_lab), y), 'float32')
        c_del = [ch.c_cpt + ch.c_fav_act for ch in self.children]
        c_act = ts.concatenate([c_err] + c_del, axis=1)
        self.c_fav_act = ts.sum(π_ts * c_act, axis=1, keepdims=True)
        self.c_cpt = self.tf.c_cpt

        # compute the local loss
        l_err = ts.sum(π_tr * ts.sqr(c_act_est - c_act), axis=1, keepdims=True)
        l_l2 = k_l2 * ts.sum(ts.sqr(self.w)) + self.tf.l_l2
        self.l_lay = l_err + l_l2

        # compute global loss
        self.mean_l_net = (
            self.l_lay +
            sum(π_tr[:, n_lab+i] * ch.mean_l_net
                for i, ch in enumerate(self.children)))
        self.mean_path_len = (
            ts.ones((x.shape[0], 1)) +
            sum(π_tr[:, n_lab+i] * ch.mean_path_len
                for i, ch in enumerate(self.children)))
        self.mean_l_lay = ts.sum(self.mean_l_net) / ts.sum(self.mean_path_len)

        # classify
        self.y_est = ts.concatenate(
            [i_fav_act] +
            [ts.eq(i_fav_act, n_lab + i) * ch.y_est + ts.neq(i_fav_act, n_lab + i) * -1
             for i, ch in enumerate(self.children)],
            axis=1)

        self.val_to_print = ts.mean(ts.sqr(c_act_est - c_act), axis=0)

class Net:
    def __init__(self, root):
        x = ts.fmatrix()
        y = ts.icol()
        k_cpt = ts.fscalar()
        k_l2 = ts.fscalar()
        ϵ = ts.fscalar()
        λ = ts.fscalar()
        root.link(x, y, k_cpt, k_l2, ϵ)
        c_avg = ts.mean(root.c_fav_act)
        l_avg = root.mean_l_lay# + 1e-8 * ts.sum(th.printing.Print()(root.val_to_print))
        self.train = th.function([x, y, k_cpt, k_l2, ϵ, λ], [l_avg, c_avg], updates=[
            (p, p - λ * ts.grad(l_avg, p))
            for p in root.params()
        ], on_unused_input='ignore')
        self.classify = th.function([x], root.y_est)
        self.root = root
