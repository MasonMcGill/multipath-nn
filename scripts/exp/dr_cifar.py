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

k_cpts = [0.0, 4e-9, 8e-9, 1.2e-8, 1.6e-8, 2e-8]
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

n_epochs = 1#25
logging_period = 1#5
batch_size = 128

λ_lrn_0 = 0.001
t_anneal = 5

################################################################################
# Network Components
################################################################################

def router(n_sinks):
    return Chain(name='Router', comps=[
        SelectPyramidTop(shape=tf_specs[-1][-1]),
        LinTrans(n_chan=router_n_chan, k_l2=k_l2, σ_w=σ_w),
        BatchNorm(), Rect(), LinTrans(n_chan=n_sinks, k_l2=k_l2)])

def pyr(*sinks):
    return Chain(
        name='ToPyramidLLN', sinks=sinks,
        router=router(len(sinks)), comps=[
            ToPyramid(n_scales=tf_specs[0].n_scales),
            MultiscaleLLN(shape0=tf_specs[0].shape0_in),
            BatchNorm()])

def rcm(i, *sinks):
    return Chain(
        name='ReConvMax', sinks=sinks,
        router=router(len(sinks)), comps=[
            MultiscaleConvMax(
                shape0=tf_specs[i].shape0_in, n_scales=tf_specs[i].n_scales,
                n_chan=tf_specs[i].n_chan, supp=conv_supp, k_l2=k_l2, σ_w=σ_w),
            BatchNorm(), Rect()])

def reg(w_cls):
    return Chain(name='LogReg', comps=[
        SelectPyramidTop(shape=tf_specs[-1][-1]),
        LinTrans(n_chan=w_cls.shape[1], k_l2=k_l2, σ_w=σ_w),
        Softmax(), SuperclassCrossEntropyError(w_cls=w_cls)])

################################################################################
# Network Constructors
################################################################################

def sr_chain(w_cls, n_tf):
    def make_net():
        root = reg(w_cls)
        for i in reversed(range(n_tf)):
            root = rcm(i, root)
        root = pyr(root)
        return SRNet(
            x0_shape=x0_shape,
            y_shape=w_cls.shape[:1],
            root=root)
    return make_net

def ds_chain(w_cls, k_cpt=0.0):
    def make_net():
        root = rcm(-1, reg(w_cls))
        for i in reversed(range(len(tf_specs) - 1)):
            root = rcm(i, reg(w_cls), root)
        root = pyr(reg(w_cls), root)
        return DSNet(
            x0_shape=x0_shape,
            y_shape=w_cls.shape[:1],
            k_cpt=k_cpt, root=root)
    return make_net

def cr_chain(w_cls, k_cpt=0.0):
    def make_net():
        root = rcm(-1, reg(w_cls))
        for i in reversed(range(len(tf_specs) - 1)):
            root = rcm(i, reg(w_cls), root)
        root = pyr(reg(w_cls), root)
        return CRNet(
            x0_shape=x0_shape,
            y_shape=w_cls.shape[:1],
            k_cpt=k_cpt, root=root)
    return make_net

def ds_tree(w_cls, k_cpt=0.0):
    return lambda: DSNet(
        x0_shape=x0_shape,
        y_shape=w_cls.shape[:1],
        k_cpt=k_cpt, root=(
            pyr(reg(w_cls),
                rcm(0, reg(w_cls),
                    rcm(1, reg(w_cls),
                        rcm(2, reg(w_cls),
                            rcm(3, reg(w_cls)),
                            rcm(3, reg(w_cls))),
                        rcm(2, reg(w_cls),
                            rcm(3, reg(w_cls)),
                            rcm(3, reg(w_cls)))),
                    rcm(1, reg(w_cls),
                        rcm(2, reg(w_cls),
                            rcm(3, reg(w_cls)),
                            rcm(3, reg(w_cls))),
                        rcm(2, reg(w_cls),
                            rcm(3, reg(w_cls)),
                            rcm(3, reg(w_cls))))))))

def cr_tree(w_cls, k_cpt=0.0):
    return lambda: CRNet(
        x0_shape=x0_shape,
        y_shape=w_cls.shape[:1],
        k_cpt=k_cpt, root=(
            pyr(reg(w_cls),
                rcm(0, reg(w_cls),
                    rcm(1, reg(w_cls),
                        rcm(2, reg(w_cls),
                            rcm(3, reg(w_cls)),
                            rcm(3, reg(w_cls))),
                        rcm(2, reg(w_cls),
                            rcm(3, reg(w_cls)),
                            rcm(3, reg(w_cls)))),
                    rcm(1, reg(w_cls),
                        rcm(2, reg(w_cls),
                            rcm(3, reg(w_cls)),
                            rcm(3, reg(w_cls))),
                        rcm(2, reg(w_cls),
                            rcm(3, reg(w_cls)),
                            rcm(3, reg(w_cls))))))))

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
         for k_cpt in k_cpts])}
