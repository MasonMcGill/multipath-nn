from collections import namedtuple

import numpy as np

from lib.data import Dataset
from lib.layer_types import (
    BatchNorm, Chain, LinTrans, MultiscaleConvMax, MultiscaleLLN, Rect,
    SelectPyramidTop, Softmax, SuperclassCrossEntropyError, ToPyramid)
from lib.net_types import CRNet, DSNet, SRNet

################################################################################
# Network Hyperparameters
################################################################################

x0_shape = (32, 32, 3)
conv_supp = 3
router_n_chan = 16

k_cpts = [0.0, 4e-9, 8e-9, 1.2e-8, 1.6e-8, 2e-8][1:]
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
    def __init__(self, shape0, w_cls):
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

def reg(w_cls, i=None):
    return (LogReg(tf_specs[0][0], w_cls) if i is None
            else LogReg(tf_specs[i][3], w_cls))

def rcm(i):
    return ReConvMax(*tf_specs[i][:3])

################################################################################
# Network Constructors
################################################################################

def sr_chain(w_cls, n_tf):
    layers = reg(w_cls, n_tf - 1),
    for i in reversed(range(n_tf)):
        layers = [rcm(i), layers]
    layers = [pyr(), layers]
    return lambda: SRNet(x0_shape, w_cls.shape[:1], {}, layers)

def ds_chain(w_cls, k_cpt=0.0):
    layers = [rcm(-1), reg(w_cls, -1)]
    for i in reversed(range(len(tf_specs) - 1)):
        layers = [rcm(i), reg(w_cls, i), layers]
    layers = [pyr(), reg(w_cls), layers]
    return lambda: DSNet(x0_shape, w_cls.shape[:1], gen_ds_router,
                         dict(k_cpt=k_cpt), layers)

def cr_chain(w_cls, k_cpt=0.0, optimistic=True):
    layers = [rcm(-1), reg(w_cls, -1)]
    for i in reversed(range(len(tf_specs) - 1)):
        layers = [rcm(i), reg(w_cls, i), layers]
    layers = [pyr(), reg(w_cls), layers]
    return lambda: CRNet(x0_shape, w_cls.shape[:1], gen_cr_router,
                         optimistic, dict(k_cpt=k_cpt), layers)

def ds_tree(w_cls, k_cpt=0.0):
    return lambda: DSNet(
        x0_shape, w_cls.shape[:1],
        gen_ds_router, dict(k_cpt=k_cpt),
        [pyr(), reg(w_cls),
            [rcm(0), reg(w_cls, 0),
                [rcm(1), reg(w_cls, 1),
                    [rcm(2), reg(w_cls, 2),
                        [rcm(3), reg(w_cls, 3)],
                        [rcm(3), reg(w_cls, 3)]],
                    [rcm(2), reg(w_cls, 2),
                        [rcm(3), reg(w_cls, 3)],
                        [rcm(3), reg(w_cls, 3)]]],
                [rcm(1), reg(w_cls, 1),
                    [rcm(2), reg(w_cls, 2),
                        [rcm(3), reg(w_cls, 3)],
                        [rcm(3), reg(w_cls, 3)]],
                    [rcm(2), reg(w_cls, 2),
                        [rcm(3), reg(w_cls, 3)],
                        [rcm(3), reg(w_cls, 3)]]]]])

def cr_tree(w_cls, k_cpt=0.0, optimistic=True):
    return lambda: CRNet(
        x0_shape, w_cls.shape[:1], gen_cr_router,
        optimistic, dict(k_cpt=k_cpt),
        [pyr(), reg(w_cls),
            [rcm(0), reg(w_cls, 0),
                [rcm(1), reg(w_cls, 1),
                    [rcm(2), reg(w_cls, 2),
                        [rcm(3), reg(w_cls, 3)],
                        [rcm(3), reg(w_cls, 3)]],
                    [rcm(2), reg(w_cls, 2),
                        [rcm(3), reg(w_cls, 3)],
                        [rcm(3), reg(w_cls, 3)]]],
                [rcm(1), reg(w_cls, 1),
                    [rcm(2), reg(w_cls, 2),
                        [rcm(3), reg(w_cls, 3)],
                        [rcm(3), reg(w_cls, 3)]],
                    [rcm(2), reg(w_cls, 2),
                        [rcm(3), reg(w_cls, 3)],
                        [rcm(3), reg(w_cls, 3)]]]]])

def attention_net(w_cls, k_cpt=0.0):
    from lib.net_types import AttentionNet
    return lambda: AttentionNet(
        x0_shape, w_cls.shape[:1], dict(k_cpt=k_cpt),
        [pyr(), rcm(0), rcm(1), rcm(2), rcm(3), reg(w_cls, 3)])

################################################################################
# Experiment Specifications
################################################################################

def class_map(m):
    return np.transpose(np.float32([
        np.equal(m, i) for i in range(max(m) + 1)]))

w_cls_cifar2 = class_map([0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
w_cls_cifar10 = class_map(list(range(10)))
w_cls_hybrid = class_map(list(range(20)))

ExpSpec = namedtuple(
    'ExperimentSpec',
    'dataset nets')

exp_specs = {
    'sr-chains': ExpSpec(
        lambda: Dataset('data/cifar-10.mat'),
        [sr_chain(w_cls_cifar2, n_tf)
         for n_tf in range(len(tf_specs) + 1)]),
    'ds-chains': ExpSpec(
        lambda: Dataset('data/cifar-10.mat'),
        [ds_chain(w_cls_cifar2, k_cpt)
         for k_cpt in k_cpts]),
    'cr-chains': ExpSpec(
        lambda: Dataset('data/cifar-10.mat'),
        [cr_chain(w_cls_cifar2, k_cpt)
         for k_cpt in k_cpts]),
    'ds-chains-em': ExpSpec(
        lambda: Dataset('data/cifar-10.mat', n_vl=1280),
        [ds_chain(w_cls_cifar2, k_cpt)
         for k_cpt in k_cpts]),
    'cr-chains-em': ExpSpec(
        lambda: Dataset('data/cifar-10.mat', n_vl=1280),
        [cr_chain(w_cls_cifar2, k_cpt)
         for k_cpt in k_cpts]),
    'ds-trees': ExpSpec(
        lambda: Dataset('data/cifar-10.mat'),
        [ds_tree(w_cls_cifar2, k_cpt)
         for k_cpt in k_cpts]),
    'cr-trees': ExpSpec(
        lambda: Dataset('data/cifar-10.mat'),
        [cr_tree(w_cls_cifar2, k_cpt)
         for k_cpt in k_cpts]),
    'sr-chains-hybrid': ExpSpec(
        lambda: Dataset('data/hybrid.mat'),
        [sr_chain(w_cls_hybrid, n_tf)
         for n_tf in range(len(tf_specs) + 1)]),
    'ds-chains-em-hybrid': ExpSpec(
        lambda: Dataset('data/hybrid.mat', n_vl=1280),
        [ds_chain(w_cls_hybrid, k_cpt)
         for k_cpt in k_cpts]),
    'ds-trees-em-hybrid': ExpSpec(
        lambda: Dataset('data/hybrid.mat', n_vl=1280),
        [ds_tree(w_cls_hybrid, k_cpt)
         for k_cpt in k_cpts]),
    'attention-nets': ExpSpec(
        lambda: Dataset('data/cifar-10.mat'),
        [attention_net(w_cls_cifar2, k_cpt)
         for k_cpt in [1e-7, 1e-8, 0.0]])}
