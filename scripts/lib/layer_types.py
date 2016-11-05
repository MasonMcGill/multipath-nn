from abc import ABCMeta
from types import SimpleNamespace as Ns

import numpy as np
import tensorflow as tf

################################################################################
# Core Layer Class
################################################################################

class Layer(metaclass=ABCMeta):
    default_hypers = Ns()

    def __init__(self, **options):
        self.name = options.pop('name', type(self).__name__)
        self.router = options.pop('router', None)
        self.sinks = options.pop('sinks', [])
        self.comps = options.pop('comps', [])
        self.hypers = Ns(**{**vars(type(self).default_hypers), **options})
        self.params = Ns()

    def link(self, x, y, mode):
        self.x = x
        self.c_err = tf.zeros(())
        self.c_mod = tf.zeros(())
        self.n_ops = tf.zeros(())

################################################################################
# The No-Op Layer
################################################################################

class NoOp(Layer):
    pass

################################################################################
# Transformation Layers
################################################################################

class LinTrans(Layer):
    default_hypers = Ns(n_chan=1, k_l2=0, σ_w=1, res=False)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ, θ = self.hypers, self.params
        n_in = np.prod(x.get_shape().as_list()[1:])
        w_eq = np.eye(n_in, ϕ.n_chan) if ϕ.res else 0
        w_shape = (n_in, ϕ.n_chan)
        w_scale = ϕ.σ_w / np.sqrt(n_in)
        θ.w = tf.Variable(w_eq + w_scale * tf.random_normal(w_shape))
        θ.b = tf.Variable(tf.zeros(ϕ.n_chan))
        self.x = tf.matmul(tf.reshape(x, (-1, n_in)), θ.w) + θ.b
        self.c_mod = ϕ.k_l2 * tf.reduce_sum(tf.square(θ.w - w_eq))
        self.n_ops = n_in * ϕ.n_chan

class Conv(Layer):
    default_hypers = Ns(n_chan=1, supp=1, k_l2=0, σ_w=1, res=False)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ, θ = self.hypers, self.params
        n_in = x.get_shape().as_list()[3]
        n_pix = np.prod(x.get_shape().as_list()[1:3])
        w_shape = (ϕ.supp, ϕ.supp, n_in, ϕ.n_chan)
        w_scale = ϕ.σ_w / ϕ.supp / np.sqrt(n_in)
        w_ident = np.float32(
            (np.arange(ϕ.supp) == ϕ.supp // 2)[:, None, None, None]
            * (np.arange(ϕ.supp) == ϕ.supp // 2)[:, None, None]
            * np.eye(n_in, ϕ.n_chan))
        w_eq = w_ident if ϕ.res else 0
        θ.w = tf.Variable(w_eq + w_scale * tf.random_normal(w_shape))
        θ.b = tf.Variable(tf.zeros(ϕ.n_chan))
        self.x = tf.nn.conv2d(x, θ.w, (1, 1, 1, 1), 'SAME') + θ.b
        self.c_mod = ϕ.k_l2 * tf.reduce_sum(tf.square(θ.w - w_eq))
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
    default_hypers = Ns(stride=1, supp=1)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ = self.hypers
        strides = (1, ϕ.stride, ϕ.stride, 1)
        k_shape = (1, ϕ.supp, ϕ.supp, 1)
        self.x = tf.nn.max_pool(x, strides, k_shape, 'SAME')

class GlobalMaxPool(Layer):
    def link(self, x, y, mode):
        super().link(x, y, mode)
        dims = tuple(x.get_shape().as_list()[1:-1])
        self.x = tf.reduce_max(x, dims)

################################################################################
# Multiscale Transformation Layers
################################################################################

def to_pyramid(x, n_scales):
    h, w = x.get_shape().as_list()[1:3]
    return [tf.image.resize_images(x, (h // 2**i, w // 2**i))
            for i in range(n_scales)]

def pack_pyramid(x):
    n_chan = x[0].get_shape()[3].value
    return tf.concat(1, [
        tf.reshape(x_i, (-1, np.prod(x_i.get_shape().as_list()[1:3]), n_chan))
        for x_i in x])

def unpack_pyramid(x, shape0):
    n_pix, n_chan = x.get_shape().as_list()[1:]
    h, w = shape0
    x_unpacked = []
    i = 0
    while i + h * w <= n_pix:
        x_unpacked.append(tf.reshape(x[:, i:i+h*w, :], (-1, h, w, n_chan)))
        i += h * w
        h //= 2
        w //= 2
    return x_unpacked

class ToPyramid(Layer):
    default_hypers = Ns(n_scales=1)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        self.x = pack_pyramid(to_pyramid(x, self.hypers.n_scales))

class MultiscaleLLN(Layer):
    default_hypers = Ns(shape0=(1, 1), σ=3, ϵ=1e-3)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ = self.hypers
        s = int(np.ceil(2 * ϕ.σ))
        u = np.linspace(-s, s, 2 * s + 1)[:, None, None, None]
        v = np.linspace(-s, s, 2 * s + 1)[:, None, None]
        k = (np.exp(-(u**2 + v**2) / (2 * ϕ.σ**2)) / (2 * np.pi * ϕ.σ**2)
             * [[0.2989], [0.5870], [0.1140]])
        x_out_unpacked = []
        for x_i in unpack_pyramid(x, ϕ.shape0):
            h, w = x_i.get_shape().as_list()[1:3]
            x_i_lum = tf.nn.conv2d(
                tf.pad(x_i, [[0, 0], [s, s], [s, s], [0, 0]]),
                k, (1, 1, 1, 1), 'SAME')[:, s:s+h, s:s+w, :]
            x_i_density = tf.nn.conv2d(
                tf.pad(tf.ones_like(x_i), [[0, 0], [s, s], [s, s], [0, 0]]),
                k, (1, 1, 1, 1), 'SAME')[:, s:s+h, s:s+w, :]
            x_out_unpacked.append(x_i / (x_i_lum / x_i_density + ϕ.ϵ))
        self.x = pack_pyramid(x_out_unpacked)

class MultiscaleConvMax(Layer):
    default_hypers = Ns(
        shape0=(1, 1), n_scales=1, n_chan=1,
        supp=1, res=False, k_l2=0, σ_w=1)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ, θ = self.hypers, self.params
        n_in = x.get_shape()[2].value
        w_shape = (ϕ.supp, ϕ.supp, n_in, ϕ.n_chan)
        w_scale = ϕ.σ_w / ϕ.supp / np.sqrt(n_in)
        w_ident = np.float32(
            (np.arange(ϕ.supp) == ϕ.supp // 2)[:, None, None, None]
            * (np.arange(ϕ.supp) == ϕ.supp // 2)[:, None, None]
            * np.eye(n_in, ϕ.n_chan))
        w_eq = w_ident if ϕ.res else 0
        θ.w_horz = tf.Variable(w_eq + w_scale * tf.random_normal(w_shape))
        θ.w_vert = tf.Variable(w_scale * tf.random_normal(w_shape))
        θ.b = tf.Variable(tf.zeros(ϕ.n_chan))
        s_in = unpack_pyramid(x, ϕ.shape0)
        s_pool = [
            tf.nn.max_pool(s, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
            for s in s_in[:-1]]
        s_out = [
            θ.b + tf.nn.conv2d(s_in[i], θ.w_horz, (1, 1, 1, 1), 'SAME')
            + (tf.nn.conv2d(s_pool[i], θ.w_vert, (1, 1, 1, 1), 'SAME')
               if -i <= len(s_pool) else 0)
            for i in range(-ϕ.n_scales, 0)]
        self.x = pack_pyramid(s_out)
        self.c_mod = ϕ.k_l2 * (
            tf.reduce_sum(tf.square(θ.w_horz - w_eq)) +
            tf.reduce_sum(tf.square(θ.w_vert)))
        self.n_ops_per_scale = [
            ϕ.n_chan * ϕ.supp**2
            * np.prod(s_in[i].get_shape().as_list()[1:])
            * (1 + int(-i <= len(s_pool)))
            for i in range(-ϕ.n_scales, 0)]
        self.n_ops = sum(self.n_ops_per_scale)

class SelectPyramidTop(Layer):
    default_hypers = Ns(shape=(1, 1))

    def link(self, x, y, mode):
        super().link(x, y, mode)
        h, w = self.hypers.shape
        n_pix, n_chan = x.get_shape().as_list()[1:]
        self.x = tf.reshape(x[:, n_pix-h*w:, :], (-1, h, w, n_chan))

################################################################################
# Regularization Layers
################################################################################

class Dropout(Layer):
    default_hypers = Ns(λ=1)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        self.x = tf.nn.dropout(x, self.hypers.λ)

class BatchNorm(Layer):
    default_hypers = Ns(d=0.9, ϵ=1e-6)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ, θ = self.hypers, self.params
        n_dim = len(x.get_shape())
        n_chan = x.get_shape().as_list()[-1]
        θ.γ = tf.Variable(tf.ones(n_chan))
        θ.β = tf.Variable(tf.zeros(n_chan))
        θ.m_avg = tf.Variable(tf.zeros(n_chan), trainable=False)
        θ.v_avg = tf.Variable(tf.ones(n_chan), trainable=False)
        def x_tr():
            m_batch, v_batch = tf.nn.moments(x, tuple(range(n_dim - 1)))
            update_m = tf.assign(θ.m_avg, ϕ.d * θ.m_avg + (1 - ϕ.d) * m_batch)
            update_v = tf.assign(θ.v_avg, ϕ.d * θ.v_avg + (1 - ϕ.d) * v_batch)
            with tf.control_dependencies([update_m, update_v]):
                return θ.γ * (x - m_batch) / tf.sqrt(v_batch + ϕ.ϵ) + θ.β
        def x_ev():
            return θ.γ * (x - θ.m_avg) / tf.sqrt(θ.v_avg + ϕ.ϵ) + θ.β
        self.x = tf.cond(tf.equal(mode, 'tr'), x_tr, x_ev)

################################################################################
# Error Layers
################################################################################

class SquaredError(Layer):
    def link(self, x, y, mode):
        super().link(x, y, mode)
        self.c_err = tf.reduce_sum(tf.square(x - y), 1)
        self.δ_cor = tf.to_float(tf.equal(
            tf.argmax(self.x, 1), tf.argmax(y, 1)))

class CrossEntropyError(Layer):
    default_hypers = Ns(ϵ=1e-6)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ = self.hypers
        n_cls = y.get_shape()[1].value
        p_cls = ϕ.ϵ / n_cls + (1 - ϕ.ϵ) * x
        self.c_err = -tf.reduce_sum(y * tf.log(p_cls), 1)
        self.δ_cor = tf.to_float(tf.equal(
            tf.argmax(self.x, 1), tf.argmax(y, 1)))

class SuperclassCrossEntropyError(Layer):
    default_hypers = Ns(w_cls=None, ϵ=1e-6)

    def link(self, x, y, mode):
        super().link(x, y, mode)
        ϕ = self.hypers
        y_sup = tf.matmul(y, ϕ.w_cls)
        n_cls = y_sup.get_shape()[1].value
        p_cls = ϕ.ϵ / n_cls + (1 - ϕ.ϵ) * x
        self.c_err = -tf.reduce_sum(y_sup * tf.log(p_cls), 1)
        self.δ_cor = tf.to_float(tf.equal(
            tf.argmax(self.x, 1), tf.argmax(y_sup, 1)))

################################################################################
# Compound Layers
################################################################################

class Chain(Layer):
    def link(self, x, y, mode):
        super().link(x, y, mode)
        for ℓ in self.comps:
            ℓ.link(x, y, mode)
            x = ℓ.x
        self.x = x
        self.c_err = sum(ℓ.c_err for ℓ in self.comps)
        self.c_mod = sum(ℓ.c_mod for ℓ in self.comps)
        self.n_ops = sum(ℓ.n_ops for ℓ in self.comps)
        if len(self.comps) > 0 and hasattr(self.comps[-1], 'δ_cor'):
            self.δ_cor = self.comps[-1].δ_cor
