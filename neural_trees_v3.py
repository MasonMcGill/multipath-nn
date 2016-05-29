import itertools
import os

os.environ['THEANO_FLAGS'] = (
    'device=cpu,floatX=float32,cast_policy=numpy+floatX,' +
    'enable_initial_driver_test=False,warn_float64=raise,optimizer=None')

import numpy as np
import numpy.random as rand
import theano as th
import theano.tensor as ts
import theano.tensor.nnet as nn

################################################################################
# Transformation Definitions
################################################################################

class ReLuTF:
    def __init__(self, n_out, w_scale):
        self.n_out = n_out
        self.w_scale = w_scale

    def link(self, n_in, x, k_cpt, k_l2):
        self.w = th.shared(np.float32(self.w_scale * rand.randn(n_in, self.n_out)))
        self.b = th.shared(np.float32(self.w_scale * np.zeros(self.n_out)))
        self.x = nn.relu(ts.dot(x, self.w) + self.b)
        self.n_ops = self.w.size.eval()
        self.c_cpt = k_cpt * ts.cast(self.n_ops, 'float32')
        self.c_l2 = k_l2 * ts.sum(ts.sqr(self.w))

    def params(self):
        return [self.w, self.b]

class ConvTF:
    def __init__(self, in_shape, w_shape, w_scale):
        self.in_shape = in_shape
        self.w_shape = w_shape
        self.w_scale = w_scale

    def link(self, n_in, x, k_cpt, k_l2):
        self.w = th.shared(np.float32(self.w_scale * rand.randn(*self.w_shape)))
        self.b = (
            th.shared(np.float32(self.w_scale * np.zeros(self.w_shape[0])))
                .dimshuffle('x', 0, 'x', 'x'))
        self.n_ops = self.w.size.eval() * np.prod(self.in_shape)
        self.n_out = (
            self.w_shape[0] *
            (self.in_shape[1] - self.w_shape[2] + 1) *
            (self.in_shape[2] - self.w_shape[3] + 1))
        x_img = ts.reshape(x, (x.shape[0],) + self.in_shape, 4)
        s_img = nn.conv2d(x_img, self.w) + self.b
        s_vec = ts.reshape(s_img, (x.shape[0], self.n_out), 2)
        self.x = nn.relu(s_vec)
        self.c_cpt = k_cpt * ts.cast(self.n_ops, 'float32')
        self.c_l2 = k_l2 * ts.sum(ts.sqr(self.w))

    def params(self):
        return [self.w, self.b]

class IdentityTF:
    def link(self, n_in, x, k_cpt, k_l2):
        self.n_out = n_in
        self.x = x
        self.c_cpt = 0
        self.c_l2 = 0
        self.n_ops = 0

    def params(self):
        return []

################################################################################
# Neural Decision Tree Definition
################################################################################

class Layer:
    def __init__(self, n_out, w_scale, tf, children):
        self.n_out = n_out
        self.w_scale = w_scale
        self.tf = tf
        self.children = children

    def params(self):
        return list(itertools.chain(
            [self.w_act, self.b_act, self.w_out, self.b_out],
             self.tf.params(), *[c.params() for c in self.children]))

    def link(self, n_in, x, y, k_cpt, k_l2, ϵ):
        # infer activity shape
        n_act = len(self.children) + 1
        n_out = self.n_out
        n_pts = x.shape[0]

        # link to the transformation
        self.tf.link(n_in, x, k_cpt, k_l2)

        # initialize parameters
        self.w_act = th.shared(np.float32(self.w_scale * rand.randn(self.tf.n_out, n_act)))
        self.b_act = th.shared(np.float32(self.w_scale * rand.randn(n_act)))
        self.w_out = th.shared(np.float32(self.w_scale * rand.randn(self.tf.n_out, n_out)))
        self.b_out = th.shared(np.float32(self.w_scale * rand.randn(n_out)))

        # propagate activity
        c_act_est = ts.dot(self.tf.x, self.w_act) + self.b_act
        i_fav_act = ts.argmin(c_act_est, axis=1, keepdims=True)
        π_ev = ts.eq(ts.arange(n_act), i_fav_act)
        # π_tr = (ϵ / n_act + (1 - ϵ) * π_ev)
        π_tr = ϵ / n_act + (1 - ϵ) * nn.softmax(-c_act_est)

        # link recursively
        for i, ch in enumerate(self.children):
            ch.link(self.tf.n_out, self.tf.x, y, k_cpt, k_l2, ϵ)

        # perform error analysis
        δ_y = ts.cast(ts.eq(y, ts.arange(n_out)), 'float32')
        δ_y_est = nn.softmax(ts.dot(self.tf.x, self.w_out) + self.b_out)
        c_out = -ts.sum(δ_y * ts.log(δ_y_est), axis=1, keepdims=True)
        c_l2 = (
            #1e-2 * k_l2 * ts.sum(ts.sqr(self.w_act)) +
            k_l2 * ts.sum(ts.sqr(self.w_act)) +
            k_l2 * ts.sum(ts.sqr(self.w_out)) +
            self.tf.c_l2)

        # self.c_tot_ev = (
        #     self.tf.c_cpt + c_l2 +
        #     π_ev[:, 0, None] * c_out +
        #     sum(π_ev[:, 1+i, None] * ch.c_tot_ev
        #         for i, ch in enumerate(self.children)))
        # c_q_est = (
        #     ts.sqr(c_act_est[:, 0, None] - c_out) +
        #     sum(ts.sqr(c_act_est[:, 1+i, None] - ch.c_tot_ev)
        #         for i, ch in enumerate(self.children)))
        self.c_tot_tr = (
            self.tf.c_cpt + c_l2 + #1e-2 * c_q_est +
            π_tr[:, 0, None] * c_out +
            sum(π_tr[:, 1+i, None] * ch.c_tot_tr
                for i, ch in enumerate(self.children)))

        # classify
        self.y_est = ts.concatenate(
            [ts.switch(i_fav_act, n_out + i_fav_act - 1,
                       ts.argmax(δ_y_est, axis=1, keepdims=True))] +
            [ts.eq(i_fav_act, 1 + i) * ch.y_est + ts.neq(i_fav_act, 1 + i) * -1
             for i, ch in enumerate(self.children)],
            axis=1)

class Net:
    def __init__(self, n_in, root):
        x = ts.fmatrix()
        y = ts.icol()
        k_cpt = ts.fscalar()
        k_l2 = ts.fscalar()
        ϵ = ts.fscalar()
        λ = ts.fscalar()
        root.link(n_in, x, y, k_cpt, k_l2, ϵ)
        c = ts.mean(root.c_tot_tr)
        self.train = th.function([x, y, k_cpt, k_l2, ϵ, λ], [c], updates=[
            (p, p - λ * ts.grad(c, p, disconnected_inputs='warn'))
            for p in root.params()
        ], on_unused_input='ignore')
        self.classify = th.function([x], root.y_est)
        self.root = root

####

# cost = (
#     sum(node.p_tr * (node.c_err + node.c_cpt + node.c_l2)
#         for node in output_nodes) +
#     sum(node.p_tr * (node.c_q_est + node.c_l2)
#         for node in routing_nodes))
#
# cost =
#     sum(node.p_tr * (node.c_q_est + node.c_l2 + node.tf.p_tr * (node.tf.c_err + node.tf.c_cpt + node.tf.c_l2))
#         for node in routing_nodes))
