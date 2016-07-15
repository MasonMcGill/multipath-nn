import numpy as np
import tensorflow as tf

################################################################################
# Network
################################################################################

class Net:
    def __init__(self, x0_shape, y_shape, root):
        self.x0 = tf.placeholder(tf.float32, (None,) + x0_shape)
        self.y = tf.placeholder(tf.float32, (None,) + y_shape)
        def link(layer, x):
            n_sinks = len(layer.sinks)
            layer.x = x
            layer.π_tr = tf.ones((tf.shape(x)[0], n_sinks)) / n_sinks
            layer.π_ev = tf.ones((tf.shape(x)[0], n_sinks)) / n_sinks
            layer.link_forward(x)
            for sink in layer.sinks:
                link(sink, layer.x)
            layer.ℓ_loc = tf.zeros(tf.shape(x)[:1])
            layer.link_backward(self.y)
            layer.ℓ_tr = (
                layer.ℓ_loc +
                sum(layer.π_tr[:, i] * s.ℓ_tr
                    for i, s in enumerate(layer.sinks)))
            layer.ℓ_ev = (
                layer.ℓ_loc +
                sum(layer.π_ev[:, i] * s.ℓ_ev
                    for i, s in enumerate(layer.sinks)))
        link(root, self.x0)
        self.root = root
        self.ℓ_tr = root.ℓ_tr
        self.ℓ_ev = root.ℓ_ev

################################################################################
# Layers
################################################################################

# layer properties:
# - forward linking:
#   - x: output activity (default: input activity)
#   - π_tr: routing policy during training (default: uniform)
#   - π_ev: routing policy during evaluation (default: uniform)
# - backward linking:
#   - ℓ_loc: layer-local loss (default: 0)

class ReLin:
    def __init__(self, n_chan, sink):
        self.n_chan = n_chan
        self.sinks = [sink]

    def link_forward(self, x):
        n_chan_in = np.prod([d.value for d in x.get_shape()[1:]])
        x_flat = tf.reshape(x, (tf.shape(x)[0], n_chan_in))
        w_shape = (n_chan_in, self.n_chan)
        w_scale = 2 / np.sqrt(n_chan_in)
        self.w = tf.Variable(w_scale * tf.random_normal(w_shape))
        self.b = tf.Variable(tf.zeros(self.n_chan))
        self.x = tf.nn.relu(tf.matmul(x_flat, self.w) + self.b)

    def link_backward(self, y):
        pass

class ReConv:
    def __init__(self, n_chan, step, supp, sink):
        self.n_chan = n_chan
        self.step = step
        self.supp = supp
        self.sinks = [sink]

    def link_forward(self, x):
        u = np.linspace(-2, 2, self.supp)[:, None, None, None]
        v = np.linspace(-2, 2, self.supp)[:, None, None]
        w_env = np.exp(-(u**2 - v**2) / 2) / np.sum(np.exp(-(u**2 - v**2) / 2))
        n_chan_in = x.get_shape()[3].value
        w_scale = w_env * np.sqrt(self.supp**2 / n_chan_in)
        w_shape = (self.supp, self.supp, n_chan_in, self.n_chan)
        steps = (1, self.step, self.step, 1)
        self.w = tf.Variable(w_scale * tf.random_normal(w_shape))
        self.b = tf.Variable(tf.zeros(self.n_chan))
        self.x = tf.nn.relu(tf.nn.conv2d(x, self.w, steps, 'SAME') + self.b)

    def link_backward(self, y):
        pass

class LogReg:
    def __init__(self, n_classes, ϵ=1e-6):
        self.n_classes = n_classes
        self.ϵ = ϵ
        self.sinks = []

    def link_forward(self, x):
        n_chan_in = np.prod([d.value for d in x.get_shape()[1:]])
        x_flat = tf.reshape(x, (tf.shape(x)[0], n_chan_in))
        w_shape = (n_chan_in, self.n_classes)
        w_scale = 1 / np.sqrt(n_chan_in)
        self.w = tf.Variable(w_scale * tf.random_normal(w_shape))
        self.b = tf.Variable(tf.zeros(self.n_classes))
        self.x = tf.nn.softmax(tf.matmul(x_flat, self.w) + self.b)

    def link_backward(self, y):
        self.ℓ_loc = -tf.reduce_sum(y * tf.log(tf.maximum(self.ϵ, self.x)), 1)

class DSRouting:
    def __init__(self, *sinks):
        self.sinks = sinks

    def link_forward(self, x):
        n_chan_in = np.prod([d.value for d in x.get_shape()[1:]])
        x_flat = tf.reshape(x, (tf.shape(x)[0], n_chan_in))
        w_shape = (n_chan_in, len(self.sinks))
        w_scale = 1 / np.sqrt(n_chan_in)
        self.w = tf.Variable(w_scale * tf.random_normal(w_shape))
        self.b = tf.Variable(tf.zeros(len(self.sinks)))
        self.π_tr = tf.nn.softmax(tf.matmul(x_flat, self.w) + self.b)
        self.π_ev = tf.to_float(
            tf.equal(self.π_tr, tf.reduce_max(self.π_tr, 1, True)))

    def link_backward(self, y):
        pass

################################################################################
# Sample Network Constructors
################################################################################

# net-name := {ds,cr}_{fc,cc}_{cascade,bintree,chianet}

def ds_fc_cascade(k_cpt=0):
    return Net((28, 28, 1), (10,),
        DSRouting(LogReg(10),
            ReLin(256, DSRouting(LogReg(10),
                ReLin(256, LogReg(10))))))

def ds_fc_bintree(k_cpt=0):
    return Net((28, 28, 1), (10,),
        DSRouting(LogReg(10),
            ReLin(256, DSRouting(LogReg(10),
                ReLin(256, LogReg(10)),
                ReLin(256, LogReg(10)))),
            ReLin(256, DSRouting(LogReg(10),
                ReLin(256, LogReg(10)),
                ReLin(256, LogReg(10))))))
