from abc import ABCMeta
from types import SimpleNamespace as Namespace

import numpy as np
import tensorflow as tf

################################################################################
# Core Layer Class
################################################################################

class Layer(metaclass=ABCMeta):
    default_hypers = {}

    def __init__(self, **hypers):
        full_hyper_dict = {**self.__class__.default_hypers, **hypers}
        self.hypers = Namespace(**full_hyper_dict)
        self.params = Namespace()

    def link(self, x, y, mode):
        self.x = x
        self.c_err = tf.zeros(())
        self.c_mod = tf.zeros(())
        self.n_ops = tf.zeros(())

################################################################################
# Transformation Layers
################################################################################

class LinTrans(Layer):
    default_hypers = dict(n_chan=1, k_l2=0, σ_w=1)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ, θ = self.hypers, self.params
        n_in = np.prod(x.get_shape().as_list()[1:])
        w_shape = (n_in, ϕ.n_chan)
        w_scale = ϕ.σ_w / np.sqrt(n_in)
        θ.w = tf.Variable(w_scale * tf.random_normal(w_shape))
        θ.b = tf.Variable(tf.zeros(ϕ.n_chan))
        self.x = tf.matmul(tf.reshape(x, (-1, n_in)), θ.w) + θ.b
        self.c_mod = ϕ.k_l2 * tf.reduce_sum(tf.square(θ.w))
        self.n_ops = n_in * ϕ.n_chan

class Conv(Layer):
    default_hypers = dict(n_chan=1, supp=1, k_l2=0, σ_w=1)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ, θ = self.hypers, self.params
        n_in = x.get_shape().as_list()[3]
        n_pix = np.prod(x.get_shape().as_list()[1:3])
        w_shape = (ϕ.supp, ϕ.supp, n_in, ϕ.n_chan)
        w_scale = ϕ.σ_w / ϕ.supp / np.sqrt(n_in)
        θ.w = tf.Variable(w_scale * tf.random_normal(w_shape))
        θ.b = tf.Variable(tf.zeros(ϕ.n_chan))
        self.x = tf.nn.conv2d(x, θ.w, (1, 1, 1, 1), 'SAME') + θ.b
        self.c_mod = ϕ.k_l2 * tf.reduce_sum(tf.square(θ.w))
        self.n_ops = n_pix * ϕ.supp**2 * n_in * ϕ.n_chan

class Rect(Layer):
    def link(self, x, y, mode):
        super().link(x, y, mode)
        self.x = tf.nn.relu(x)

class Softmax(Layer):
    def link(self, x, y, mode):
        super().link(x, y, mode)
        self.x = tf.nn.softmax(x)

class MaxPool(Layer):
    default_hypers = dict(stride=1, supp=1)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ = self.hypers
        strides = (1, ϕ.stride, ϕ.stride, 1)
        k_shape = (1, ϕ.supp, ϕ.supp, 1)
        self.x = tf.nn.max_pool(x, strides, k_shape, 'SAME')

################################################################################
# Regularization Layers
################################################################################

class Dropout(Layer):
    default_hypers = dict(λ=1)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        self.x = tf.nn.dropout(x, self.hypers.λ)

class BatchNorm(Layer):
    default_hypers = dict(d=0.9, ϵ=1e-6)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ, θ = self.hypers, self.params
        n_dim = len(x.get_shape())
        n_chan = x.get_shape().as_list()[-1]
        θ.γ = tf.Variable(tf.ones(n_chan))
        θ.β = tf.Variable(tf.zeros(n_chan))
        θ.m_avg = tf.Variable(tf.zeros(n_chan), trainable=False)
        θ.v_avg = tf.Variable(tf.ones(n_chan), trainable=False)
        m_batch, v_batch = tf.nn.moments(x, tuple(range(n_dim - 1)))
        update_m = tf.assign(θ.m_avg, ϕ.d * θ.m_avg + (1 - ϕ.d) * m_batch)
        update_v = tf.assign(θ.v_avg, ϕ.d * θ.v_avg + (1 - ϕ.d) * v_batch)
        with tf.control_dependencies([update_m, update_v]):
            x_tr = θ.γ * (x - m_batch) / tf.sqrt(v_batch + ϕ.ϵ) + θ.β
        x_ev = θ.γ * (x - θ.m_avg) / tf.sqrt(θ.v_avg + ϕ.ϵ) + θ.β
        self.x = tf.cond(tf.equal(mode, 'tr'), lambda: x_tr, lambda: x_ev)

################################################################################
# Error Layers
################################################################################

class SquaredError(Layer):
    def link(self, x, y, mode):
        super().link(x, y, mode)
        self.c_err = tf.reduce_sum(tf.square(x - y), 1)

class CrossEntropyError(Layer):
    default_hypers = dict(ϵ=1e-6)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ = self.hypers
        n_cls = y.get_shape()[1].value
        p_cls = ϕ.ϵ / n_cls + (1 - ϕ.ϵ) * x
        self.c_err = -tf.reduce_sum(y * tf.log(p_cls), 1)

################################################################################
# Compound Layers
################################################################################

class Chain(Layer):
    def __init__(self, *comps):
        super().__init__(comps=[type(c).__name__ for c in comps])
        self.comps = comps

    def link(self, x, y, mode):
        super().link(x, y, mode)
        for ℓ in self.comps:
            ℓ.link(x, y, mode)
            x = ℓ.x
        self.x = x
        self.c_err = sum(ℓ.c_err for ℓ in self.comps)
        self.c_mod = sum(ℓ.c_mod for ℓ in self.comps)
        self.n_ops = sum(ℓ.n_ops for ℓ in self.comps)
        self.params = Namespace(**{
            ('layer%i_%s' % (i, k)): v
            for i, ℓ in enumerate(self.comps)
            for k, v in vars(ℓ.params).items()})
