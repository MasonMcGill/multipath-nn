from collections import namedtuple

from lib.layer_types import (
    BatchNorm, Chain, LinTrans, MultiscaleConvMax, MultiscaleLLN, Rect,
    SelectPyramidTop, Softmax, SquaredError, ToPyramid)
from lib.net_types import CRNet, DSNet, SRNet

################################################################################
# Network Hyperparameters
################################################################################

x0_shape = (32, 32, 3)
y_shape = (2,)

conv_supp = 3
router_n_chan = 16

k_cpts = [0, 1e-9, 2e-9, 4e-9, 8e-9, 1.6e-8]
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
            LinTrans(n_chan=y_shape[0], k_l2=k_l2, σ_w=σ_w),
            Softmax(), SquaredError())

def gen_router(ℓ):
    return Chain(
        SelectPyramidTop(shape=tf_specs[-1][-1]),
        LinTrans(n_chan=router_n_chan, k_l2=k_l2, σ_w=σ_w),
        BatchNorm(), Rect(), LinTrans(n_chan=len(ℓ.sinks), k_l2=k_l2))

################################################################################
# Network Constructors
################################################################################

def sr_chain(n_tf):
    layers = LogReg(tf_specs[n_tf-1][0])
    for spec in reversed(tf_specs[:n_tf]):
        layers = [ReConvMax(*spec[:3]), layers]
    layers = [ToPyramidLLN(*tf_specs[0][:2]), layers]
    return SRNet(x0_shape, y_shape, layers)

def ds_chain():
    layers = [ReConvMax(*tf_specs[-1][:3]), LogReg(tf_specs[-1][0])]
    for spec in reversed(tf_specs[:-1]):
        layers = [ReConvMax(*spec[:3]), LogReg(spec[3]), layers]
    layers = [ToPyramidLLN(*tf_specs[0][:2]), LogReg(tf_specs[0][3]), layers]
    return DSNet(x0_shape, y_shape, gen_router, layers)

def cr_chain(optimistic=True):
    layers = [ReConvMax(*tf_specs[-1][:3]), LogReg(tf_specs[-1][0])]
    for spec in reversed(tf_specs[:-1]):
        layers = [ReConvMax(*spec[:3]), LogReg(spec[3]), layers]
    layers = [ToPyramidLLN(*tf_specs[0][:2]), LogReg(tf_specs[0][3]), layers]
    return CRNet(x0_shape, y_shape, gen_router, optimistic, layers)
