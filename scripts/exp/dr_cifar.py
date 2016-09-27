from collections import namedtuple

import numpy as np

from lib.data import Dataset
from lib.layer_types import (
    BatchNorm, Chain, LinTrans, MultiscaleConvMax, MultiscaleLLN, Rect,
    SelectPyramidTop, Softmax, SuperclassCrossEntropyError, ToPyramid)
from lib.net_types import CRNet, DSNet, SRNet

################################################################################
# Dataset and  Subclass -> Superclass Mappings
################################################################################

def read_dataset():
    return Dataset('data/cifar-10.mat', n_vl=1280)

x0_shape = (32, 32, 3)
y_shape = (10,)

m_cls = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
w_cls = np.transpose(np.float32([np.equal(m_cls, i) for i in range(2)]))

################################################################################
# Network Hyperparameters
################################################################################

conv_supp = 3
router_n_chan = 16

k_cpts = [0, 4e-9, 8e-9, 1.2e-8, 1.6e-8, 2e-8]
k_cre = 0.01
k_l2 = 1e-3
σ_w = 1e-2

TFSpec = namedtuple(
    'TransformSpec',
    'shape0_in n_scales '
    'n_chan shape0_out')

tf_specs = [
    TFSpec((32, 32), 4, 32, (32, 32)),
    TFSpec((32, 32), 4, 32, (32, 32)),
    TFSpec((32, 32), 3, 32, (16, 16)),
    TFSpec((16, 16), 3, 64, (16, 16)),
    TFSpec((16, 16), 3, 64, (16, 16)),
    TFSpec((16, 16), 2, 64, (8, 8)),
    TFSpec((8, 8), 2, 128, (8, 8)),
    TFSpec((8, 8), 2, 128, (8, 8)),
    TFSpec((8, 8), 1, 128, (4, 4)),
    TFSpec((4, 4), 1, 256, (4, 4)),
    TFSpec((4, 4), 1, 256, (4, 4)),
    TFSpec((4, 4), 1, 256, (4, 4))]

################################################################################
# Training Hyperparameters
################################################################################

n_epochs = 25
logging_period = 5
batch_size = 128

λ_lrn_0 = 0.001
t_anneal = 5

################################################################################
# Network Components
################################################################################

class ToPyramidLLN(Chain):
    def __init__(self, shape0, n_scales):
        super().__init__(
            ToPyramid(n_scales=n_scales),
            MultiscaleLLN(shape0=shape0),
            BatchNorm())

class ReConvMax(Chain):
    def __init__(self, shape0, n_scales, n_chan):
        super().__init__(
            MultiscaleConvMax(
                shape0=shape0, n_scales=n_scales, n_chan=n_chan,
                supp=conv_supp, k_l2=k_l2, σ_w=σ_w),
            BatchNorm(), Rect())

class LogReg(Chain):
    def __init__(self, shape0):
        super().__init__(
            SelectPyramidTop(shape=tf_specs[-1][-1]),
            LinTrans(n_chan=w_cls.shape[1], k_l2=k_l2, σ_w=σ_w),
            Softmax(), SuperclassCrossEntropyError(w_cls=w_cls))

def gen_ds_router(ℓ):
    return Chain(
        SelectPyramidTop(shape=tf_specs[-1][-1]),
        LinTrans(n_chan=router_n_chan, k_l2=k_l2, σ_w=σ_w),
        BatchNorm(), Rect(), LinTrans(n_chan=len(ℓ.sinks), k_l2=k_l2))

def gen_cr_router(ℓ):
    return Chain(
        SelectPyramidTop(shape=tf_specs[-1][-1]),
        LinTrans(n_chan=router_n_chan, k_l2=(k_l2 * k_cre), σ_w=σ_w),
        BatchNorm(), Rect(), LinTrans(n_chan=len(ℓ.sinks), k_l2=(k_l2 * k_cre)))

################################################################################
# Layer Tree Construction Shorthand
################################################################################

def pyr():
    return ToPyramidLLN(*tf_specs[0][:2])

def reg(i=None):
    return LogReg(tf_specs[0][0]) if i is None else LogReg(tf_specs[i][3])

def rcm(i):
    return ReConvMax(*tf_specs[i][:3])

################################################################################
# Network Constructors
################################################################################

def sr_chain(n_tf):
    layers = reg(n_tf - 1),
    for i in reversed(range(n_tf)):
        layers = [rcm(i), layers]
    layers = [pyr(), layers]
    return SRNet(x0_shape, y_shape, layers)

def ds_chain():
    layers = [rcm(-1), reg(-1)]
    for i in reversed(range(len(tf_specs) - 1)):
        layers = [rcm(i), reg(i), layers]
    layers = [pyr(), reg(), layers]
    return DSNet(x0_shape, y_shape, gen_ds_router, layers)

def cr_chain(optimistic=True):
    layers = [rcm(-1), reg(-1)]
    for i in reversed(range(len(tf_specs) - 1)):
        layers = [rcm(i), reg(i), layers]
    layers = [pyr(), reg(), layers]
    return CRNet(x0_shape, y_shape, gen_cr_router,
                 optimistic, layers)

def ds_tree():
    return DSNet(
        x0_shape, y_shape, gen_ds_router,
        [pyr(), reg(),
            [rcm(0), reg(0),
                [rcm(1), reg(1),
                    [rcm(2), reg(2),
                        [rcm(3), reg(3)],
                        [rcm(3), reg(3)]],
                    [rcm(2), reg(2),
                        [rcm(3), reg(3)],
                        [rcm(3), reg(3)]]],
                [rcm(1), reg(1),
                    [rcm(2), reg(2),
                        [rcm(3), reg(3)],
                        [rcm(3), reg(3)]],
                    [rcm(2), reg(2),
                        [rcm(3), reg(3)],
                        [rcm(3), reg(3)]]]]])

def cr_tree(optimistic=True):
    return CRNet(
        x0_shape, y_shape,
        gen_cr_router, optimistic,
        [pyr(), reg(),
            [rcm(0), reg(0),
                [rcm(1), reg(1),
                    [rcm(2), reg(2),
                        [rcm(3), reg(3)],
                        [rcm(3), reg(3)]],
                    [rcm(2), reg(2),
                        [rcm(3), reg(3)],
                        [rcm(3), reg(3)]]],
                [rcm(1), reg(1),
                    [rcm(2), reg(2),
                        [rcm(3), reg(3)],
                        [rcm(3), reg(3)]],
                    [rcm(2), reg(2),
                        [rcm(3), reg(3)],
                        [rcm(3), reg(3)]]]]])
