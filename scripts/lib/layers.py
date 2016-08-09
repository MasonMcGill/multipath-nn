from abc import ABCMeta
from types import SimpleNamespace as Namespace

import numpy as np
import tensorflow as tf

################################################################################
# Root Layer Class
################################################################################

class Layer(metaclass=ABCMeta):
    default_params = {}

    def __init__(self, params, *sinks):
        full_param_dict = {**self.__class__.default_params, **params}
        self.params = Namespace(**full_param_dict)
        self.sinks = sinks

    def link(self, sigs):
        self.x = sigs.x
        self.c_err = 0
        self.c_mod = 0
        self.n_ops = 0

################################################################################
# Regression Layers
################################################################################

class LogReg(Layer):
    default_params = dict(k_l2=0, ϵ=1e-6)

    def link(self, sigs):
        super().link(sigs)
        k_l2, ϵ = self.params.k_l2, self.params.ϵ
        n_cls = sigs.y.get_shape()[1].value
        n_chan_in = np.prod(sigs.x.get_shape().as_list()[1:])
        w_shape = (n_chan_in, n_cls)
        w_scale = 1 / np.sqrt(n_chan_in)
        w = tf.Variable(w_scale * tf.random_normal(w_shape))
        b = tf.Variable(tf.zeros(n_cls))
        x_flat = tf.reshape(sigs.x, (-1, n_chan_in))
        self.x = tf.nn.softmax(tf.matmul(x_flat, w) + b)
        p_cls = ϵ / n_cls + (1 - ϵ) * self.x
        self.c_err = -tf.reduce_sum(sigs.y * tf.log(p_cls), 1)
        self.c_mod = k_l2 * tf.reduce_sum(tf.square(w))

################################################################################
# Transformation Layers
################################################################################

class LinTrans(Layer):
    default_params = dict(n_chan=1, k_l2=0)

    def link(self, sigs):
        super().link(sigs)
        n_chan, k_l2 = self.params.n_chan, self.params.k_l2
        n_chan_in = np.prod(sigs.x.get_shape().as_list()[1:])
        w_shape = (n_chan_in, n_chan)
        w_scale = 1 / np.sqrt(n_chan_in)
        w = tf.Variable(w_scale * tf.random_normal(w_shape))
        b = tf.Variable(tf.zeros(n_chan))
        x_flat = tf.reshape(sigs.x, (-1, n_chan_in))
        self.x = tf.matmul(x_flat, w) + b
        self.c_mod = k_l2 * tf.reduce_sum(tf.square(w))
        self.n_ops = np.prod(w.get_shape().as_list())

class Conv(Layer):
    default_params = dict(n_chan=2, supp=1, k_l2=0)

    def link(self, sigs):
        super().link(sigs)
        n_chan = self.params.n_chan
        supp = self.params.supp
        k_l2 = self.params.k_l2
        n_pix = np.prod(sigs.x.get_shape().as_list()[1:3])
        n_chan_in = sigs.x.get_shape()[3].value
        w_shape = (supp, supp, n_chan_in, n_chan)
        w_scale = 1 / np.sqrt(supp**2 * n_chan_in)
        w = tf.Variable(w_scale * tf.random_normal(w_shape))
        b = tf.Variable(tf.zeros(n_chan))
        self.x = tf.nn.conv2d(sigs.x, w, (1, 1, 1, 1), 'SAME') + b
        self.c_mod = k_l2 * tf.reduce_sum(tf.square(w))
        self.n_ops = np.prod(w.get_shape().as_list()) * n_pix

class Rect(Layer):
    def link(self, sigs):
        super().link(sigs)
        self.x = tf.nn.relu(sigs.x)

class MaxPool(Layer):
    default_params = dict(stride=1, supp=1)

    def link(self, sigs):
        super().link(sigs)
        stride, supp = self.params.stride, self.params.supp
        stride_spec, supp_spec = (1, stride, stride, 1), (1, supp, supp, 1)
        self.x = tf.nn.max_pool(sigs.x, supp_spec, stride_spec, 'SAME')

################################################################################
# Regularization Layers
################################################################################

class Dropout(Layer):
    default_params = dict(λ=1)

    def link(self, sigs):
        super().link(sigs)
        if self.params.λ != 1:
            mask = tf.less(tf.random_uniform(tf.shape(sigs.x)), self.params.λ)
            x_drop = tf.to_float(mask) * sigs.x / self.params.λ
            self.x = tf.cond(tf.equal(sigs.mode, 'tr'),
                lambda: x_drop, lambda: sigs.x)

class BatchNorm(Layer):
    default_params = dict(d=0.9, ϵ=1e-6)

    def link(self, sigs):
        super().link(sigs)
        d, ϵ = self.params.d, self.params.ϵ
        n_dim = len(sigs.x.get_shape())
        n_chan = sigs.x.get_shape()[-1].value
        γ = tf.Variable(tf.ones(n_chan))
        β = tf.Variable(tf.zeros(n_chan))
        m_avg = tf.Variable(tf.zeros(n_chan), trainable=False)
        v_avg = tf.Variable(tf.ones(n_chan), trainable=False)
        m_batch, v_batch = tf.nn.moments(sigs.x, tuple(range(n_dim - 1)))
        update_m = tf.assign(m_avg, d * m_avg + (1 - d) * m_batch)
        update_v = tf.assign(v_avg, d * v_avg + (1 - d) * v_batch)
        with tf.control_dependencies([update_m, update_v]):
            x_tr = γ * (sigs.x - m_batch) / tf.sqrt(v_batch + ϵ) + β
        x_ev = γ * (sigs.x - m_avg) / tf.sqrt(v_avg + ϵ) + β
        self.x = tf.cond(tf.equal(sigs.mode, 'tr'),
            lambda: x_tr, lambda: x_ev)
