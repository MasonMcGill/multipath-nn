import numpy as np
import tensorflow as tf

################################################################################
# Network
################################################################################

class Net:
    def __init__(self, x0_shape, y_shape, root):
        self.x0 = tf.placeholder(tf.float32, (None,) + x0_shape)
        self.y = tf.placeholder(tf.float32, (None,) + y_shape)
        root.p_tr = tf.ones(tf.shape(self.x0)[:1])
        root.p_ev = tf.ones(tf.shape(self.x0)[:1])
        def link(layer, x_tr, x_ev):
            n_sinks = len(layer.sinks)
            layer.x_tr = x_tr
            layer.x_ev = x_ev
            layer.π_tr = tf.ones((tf.shape(x_tr)[0], n_sinks)) / n_sinks
            layer.π_ev = tf.ones((tf.shape(x_ev)[0], n_sinks)) / n_sinks
            layer.n_ops = tf.zeros(())
            layer.link_forward(x_tr, x_ev)
            for i, sink in enumerate(layer.sinks):
                sink.p_tr = layer.p_tr * layer.π_tr[:, i]
                sink.p_ev = layer.p_ev * layer.π_ev[:, i]
                link(sink, layer.x_tr, layer.x_ev)
            layer.λ_tr = (
                layer.p_tr
                + sum(layer.π_tr[:, i] * s.λ_tr
                      for i, s in enumerate(layer.sinks)))
            layer.λ_ev = (
                layer.p_tr
                + sum(layer.π_ev[:, i] * s.λ_ev
                      for i, s in enumerate(layer.sinks)))
            layer.ℓℓ_tr = tf.zeros(tf.shape(x_tr)[:1])
            layer.ℓℓ_ev = tf.zeros(tf.shape(x_ev)[:1])
            layer.link_backward(self.y)
            layer.ℓ_tr = (
                layer.ℓℓ_tr
                + sum(layer.π_tr[:, i] * s.ℓ_tr
                      for i, s in enumerate(layer.sinks)))
            layer.ℓ_ev = (
                layer.ℓℓ_ev
                + sum(layer.π_ev[:, i] * s.ℓ_ev
                      for i, s in enumerate(layer.sinks)))
        link(root, self.x0, self.x0)
        self.root = root
        self.ℓ_tr = root.ℓ_tr
        self.ℓ_ev = root.ℓ_ev

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
# Regularization Functions
################################################################################

def drop(x, λ):
    mask = tf.less(tf.random_uniform(tf.shape(x)), λ)
    return x if λ == 1 else tf.to_float(mask) * x / λ

################################################################################
# Regression Layers
################################################################################

class LogReg:
    def __init__(self, n_classes, ϵ=1e-6):
        self.n_classes = n_classes
        self.ϵ = ϵ
        self.sinks = []

    def link_forward(self, x_tr, x_ev):
        n_chan_in = np.prod([d.value for d in x_tr.get_shape()[1:]])
        x_tr_flat = tf.reshape(x_tr, (tf.shape(x_tr)[0], n_chan_in))
        x_ev_flat = tf.reshape(x_ev, (tf.shape(x_ev)[0], n_chan_in))
        w_shape = (n_chan_in, self.n_classes)
        w_scale = 1 / np.sqrt(n_chan_in)
        self.w = tf.Variable(w_scale * tf.random_normal(w_shape))
        self.b = tf.Variable(tf.zeros(self.n_classes))
        self.x_tr = tf.nn.softmax(tf.matmul(x_tr_flat, self.w) + self.b)
        self.x_ev = tf.nn.softmax(tf.matmul(x_ev_flat, self.w) + self.b)

    def link_backward(self, y):
        p_cls = self.ϵ / self.n_classes + (1 - self.ϵ) * self.x_tr
        ℓ_err = -tf.reduce_sum(y * tf.log(p_cls), 1)
        self.ℓℓ_tr = ℓ_err
        self.ℓℓ_ev = ℓ_err

################################################################################
# Transformation Layers
################################################################################

class ReLin:
    def __init__(self, n_chan, k_cpt, λ, sink):
        self.n_chan = n_chan
        self.k_cpt = k_cpt
        self.λ = λ
        self.sinks = [sink]

    def link_forward(self, x_tr, x_ev):
        n_chan_in = np.prod([d.value for d in x_tr.get_shape()[1:]])
        w_shape = (n_chan_in, self.n_chan)
        w_scale = 1 / np.sqrt(n_chan_in)
        self.w = tf.Variable(w_scale * tf.random_normal(w_shape))
        self.b = tf.Variable(tf.zeros(self.n_chan))
        x_tr_flat = tf.reshape(x_tr, (tf.shape(x_tr)[0], n_chan_in))
        x_ev_flat = tf.reshape(x_ev, (tf.shape(x_ev)[0], n_chan_in))
        s_tr = tf.matmul(x_tr_flat, self.w) + self.b
        s_ev = tf.matmul(x_ev_flat, self.w) + self.b
        self.x_tr = drop(tf.nn.relu(s_tr), self.λ)
        self.x_ev = tf.nn.relu(s_ev)
        self.n_ops = np.prod(self.w.get_shape().as_list())

    def link_backward(self, y):
        ℓ_cpt = self.k_cpt * self.n_ops
        self.ℓℓ_tr = ℓ_cpt
        self.ℓℓ_ev = ℓ_cpt

class ReConv:
    def __init__(self, n_chan, step, supp, k_cpt, λ, sink):
        self.n_chan = n_chan
        self.step = step
        self.supp = supp
        self.k_cpt = k_cpt
        self.λ = λ
        self.sinks = [sink]

    def link_forward(self, x_tr, x_ev):
        u = np.linspace(-2, 2, self.supp)[:, None, None, None]
        v = np.linspace(-2, 2, self.supp)[:, None, None]
        w_env = np.exp(-(u**2 - v**2) / 2) / np.sum(np.exp(-(u**2 - v**2) / 2))
        n_chan_in = x_tr.get_shape()[3].value
        w_scale = w_env * np.sqrt(self.supp**2 / n_chan_in)
        w_shape = (self.supp, self.supp, n_chan_in, self.n_chan)
        steps = (1, self.step, self.step, 1)
        self.w = tf.Variable(w_scale * tf.random_normal(w_shape))
        self.b = tf.Variable(tf.zeros(self.n_chan))
        s_tr = tf.nn.conv2d(x_tr, self.w, steps, 'SAME') + self.b
        s_ev = tf.nn.conv2d(x_ev, self.w, steps, 'SAME') + self.b
        self.x_tr = drop(tf.nn.relu(s_tr), self.λ)
        self.x_ev = tf.nn.relu(s_ev)
        n_px = np.prod(self.x_tr.get_shape().as_list()[1:3])
        self.n_ops = np.prod(self.w.get_shape().as_list()) * n_px / self.step**2

    def link_backward(self, y):
        ℓ_cpt = self.k_cpt * self.n_ops
        self.ℓℓ_tr = ℓ_cpt
        self.ℓℓ_ev = ℓ_cpt

class ReConvMP:
    def __init__(self, n_chan, step, supp, k_cpt, λ, sink):
        self.n_chan = n_chan
        self.step = step
        self.supp = supp
        self.k_cpt = k_cpt
        self.λ = λ
        self.sinks = [sink]

    def link_forward(self, x_tr, x_ev):
        u = np.linspace(-2, 2, self.supp)[:, None, None, None]
        v = np.linspace(-2, 2, self.supp)[:, None, None]
        w_env = np.exp(-(u**2 - v**2) / 2) / np.sum(np.exp(-(u**2 - v**2) / 2))
        n_chan_in = x_tr.get_shape()[3].value
        w_scale = w_env * np.sqrt(self.supp**2 / n_chan_in) / self.step**2
        w_shape = (self.supp, self.supp, n_chan_in, self.n_chan)
        self.w = tf.Variable(w_scale * tf.random_normal(w_shape))
        self.b = tf.Variable(tf.zeros(self.n_chan))
        s_tr = tf.nn.conv2d(x_tr, self.w, (1, 1, 1, 1), 'SAME') + self.b
        s_ev = tf.nn.conv2d(x_ev, self.w, (1, 1, 1, 1), 'SAME') + self.b
        self.x_tr = drop(
            tf.nn.max_pool(
                tf.nn.relu(s_tr),
                (1, self.step, self.step, 1),
                (1, self.step, self.step, 1),
                'SAME'), self.λ)
        self.x_ev = tf.nn.max_pool(
            tf.nn.relu(s_tr),
            (1, self.step, self.step, 1),
            (1, self.step, self.step, 1),
            'SAME')
        n_px = np.prod(self.x_tr.get_shape().as_list()[1:3])
        self.n_ops = np.prod(self.w.get_shape().as_list()) * n_px

    def link_backward(self, y):
        ℓ_cpt = self.k_cpt * self.n_ops
        self.ℓℓ_tr = ℓ_cpt
        self.ℓℓ_ev = ℓ_cpt

################################################################################
# Routing Layers
################################################################################

class DSRouting:
    def __init__(self, ϵ, *sinks):
        self.ϵ = ϵ
        self.sinks = sinks

    def link_forward(self, x_tr, x_ev):
        n_chan_in = np.prod([d.value for d in x_tr.get_shape()[1:]])
        w_shape = (n_chan_in, len(self.sinks))
        self.w = tf.Variable(tf.random_normal(w_shape) / np.sqrt(n_chan_in))
        self.b = tf.Variable(tf.zeros(len(self.sinks)))
        x_tr_flat = tf.reshape(x_tr, (tf.shape(x_tr)[0], n_chan_in))
        x_ev_flat = tf.reshape(x_ev, (tf.shape(x_ev)[0], n_chan_in))
        s_tr = tf.matmul(x_tr_flat, self.w) + self.b
        s_ev = tf.matmul(x_ev_flat, self.w) + self.b
        self.π_tr = (
            self.ϵ / len(self.sinks)
            + (1 - self.ϵ) * tf.nn.softmax(s_tr))
        self.π_ev = tf.to_float(tf.equal(
            tf.expand_dims(tf.to_int32(tf.argmax(self.s_ev, 1)), 1),
            tf.range(len(self.sinks))))

    def link_backward(self, y):
        pass

class CRRouting:
    def __init__(self, k_cre, ϵ, *sinks):
        self.k_cre = k_cre
        self.ϵ = ϵ
        self.sinks = sinks

    def link_forward(self, x_tr, x_ev):
        n_chan_in = np.prod([d.value for d in x_tr.get_shape()[1:]])
        w_shape = (n_chan_in, len(self.sinks))
        self.w = tf.Variable(tf.random_normal(w_shape) / np.sqrt(n_chan_in))
        self.b = tf.Variable(tf.zeros(len(self.sinks)))
        x_tr_flat = tf.reshape(x_tr, (tf.shape(x_tr)[0], n_chan_in))
        x_ev_flat = tf.reshape(x_ev, (tf.shape(x_ev)[0], n_chan_in))
        self.ℓ_est_tr = tf.matmul(x_tr_flat, self.w) + self.b
        self.ℓ_est_ev = tf.matmul(x_ev_flat, self.w) + self.b
        self.π_tr = (
            self.ϵ / len(self.sinks)
            + (1 - self.ϵ) * tf.to_float(
                tf.equal(self.ℓ_est_tr, tf.reduce_min(self.ℓ_est_tr, 1, True))))
        self.π_ev = tf.to_float(tf.equal(
            tf.expand_dims(tf.to_int32(tf.argmin(self.ℓ_est_ev, 1)), 1),
            tf.range(len(self.sinks))))

    def link_backward(self, y):
        self.ℓℓ_tr = self.k_cre * sum(
            tf.square(self.sinks[i].ℓ_ev - self.ℓ_est_tr[:, i])
            for i in range(len(self.sinks)))

################################################################################
# Smart Routing Layers (to-do: clean up)
################################################################################

# class SmartDSRouting:
#     def __init__(self, ϵ, *sinks):
#         self.ϵ = ϵ
#         self.sinks = sinks
#
#     def link_forward(self, x, mode):
#         n_chan_in = np.prod([d.value for d in x.get_shape()[1:]])
#         x_flat = tf.reshape(x, (tf.shape(x)[0], n_chan_in))
#         self.w0 = tf.Variable(tf.random_normal((n_chan_in, 16)) / np.sqrt(n_chan_in))
#         self.w1 = tf.Variable(tf.random_normal((16, len(self.sinks))) / 4)
#         self.b0 = tf.Variable(tf.zeros(16))
#         self.b1 = tf.Variable(tf.zeros(len(self.sinks)))
#         self.π_tr = (
#             self.ϵ / len(self.sinks) +
#             (1 - self.ϵ) *
#             tf.nn.softmax(
#                 tf.matmul(
#                     tf.nn.relu(tf.matmul(x_flat, self.w0) + self.b0),
#                     self.w1)
#                 + self.b1))
#         self.π_ev = tf.to_float(tf.equal(
#             tf.expand_dims(tf.to_int32(tf.argmax(self.π_tr, 1)), 1),
#             tf.range(len(self.sinks))))
#
#     def link_backward(self, y):
#         pass
#
# class SmartCRRouting:
#     def __init__(self, k_cre, ϵ, *sinks):
#         self.k_cre = k_cre
#         self.ϵ = ϵ
#         self.sinks = sinks
#
#     def link_forward(self, x, mode):
#         n_chan_in = np.prod([d.value for d in x.get_shape()[1:]])
#         x_flat = tf.reshape(x, (tf.shape(x)[0], n_chan_in))
#         self.w0 = tf.Variable(tf.random_normal((n_chan_in, 16)) / np.sqrt(n_chan_in))
#         self.w1 = tf.Variable(tf.random_normal((16, len(self.sinks))) / 4)
#         self.b0 = tf.Variable(tf.zeros(16))
#         self.b1 = tf.Variable(tf.zeros(len(self.sinks)))
#         self.ℓ_est = self.b1 + tf.matmul(
#             tf.nn.relu(self.b0 + tf.matmul(x_flat, self.w0)),
#             self.w1)
#         self.π_tr = (
#             self.ϵ / len(self.sinks)
#             + (1 - self.ϵ) * tf.to_float(
#                 tf.equal(self.ℓ_est, tf.reduce_min(self.ℓ_est, 1, True))))
#         self.π_ev = tf.to_float(tf.equal(
#             tf.expand_dims(tf.to_int32(tf.argmin(self.ℓ_est, 1)), 1),
#             tf.range(len(self.sinks))))
#
#     def link_backward(self, y):
#         self.ℓℓ_tr = self.p_tr * self.k_cre * sum(
#             tf.square(self.sinks[i].ℓ_ev - self.ℓ_est[:, i])
#             for i in range(len(self.sinks)))
