from lib.layer_types import (
    BatchNorm, Chain, CrossEntropyError, LinTrans, MultiscaleBatchNorm,
    MultiscaleConvMax, MultiscaleLLN, MultiscaleRect, Rect, Select,
    Softmax, ToPyramid)

from lib.net_types import CRNet, DSNet, SRNet

################################################################################
# Network Hyperparameter.
################################################################################

conv_supp = 3
router_n_chan = 16

k_cpts = [0.0, 1e-9, 2e-9, 4e-9, 8e-9, 1.6e-8, 3.2e-8, 6.4e-8]
k_l2 = 1e-4
σ_w = 1

x0_shape = (32, 32, 3)
y_shape = (10,)

arch = [
    [16, 16, 16, 16],
    [16, 16, 16, 16],
    [32, 32, 32],
    [32, 32, 32],
    [64, 64],
    [64, 64],
    [128],
    [128]]

################################################################################
# Training Hyperparameters
################################################################################

n_iter = 80000
t_log = 2500
batch_size = 128

λ_lrn = lambda t: 0.1 / 2**(t / 10000)
τ_cr = lambda t: 0.1 / 2**(t / 20000)
τ_ds = lambda t: 1 / 2**(t / 20000)

################################################################################
# Network Components
################################################################################

def router(n_sinks):
    return None if n_sinks < 2 else Chain(name='Router', comps=[
        Select(i=-1), LinTrans(n_chan=router_n_chan, k_l2=k_l2, σ_w=σ_w),
        BatchNorm(), Rect(), LinTrans(n_chan=n_sinks, k_l2=k_l2, σ_w=0)])

def pyr(*sinks):
    return Chain(
        name='ToPyramid', sinks=sinks,
        router=router(len(sinks)), comps=[
            ToPyramid(n_scales=len(arch[0]))])

def rcm(i, *sinks):
    return Chain(
        name='ReConvMax', sinks=sinks,
        router=router(len(sinks)), comps=[
            MultiscaleConvMax(
                n_chan=arch[i], supp=conv_supp,
                k_l2=k_l2, σ_w=σ_w),
            MultiscaleBatchNorm(), MultiscaleRect()])

def reg():
    return Chain(name='LogReg', comps=[
        Select(i=-1),
        LinTrans(n_chan=y_shape[0], k_l2=k_l2, σ_w=σ_w),
        Softmax(), CrossEntropyError()])

################################################################################
# Network Constructors
################################################################################

def sr_chain(n_tf):
    def make_net():
        root = reg()
        for i in reversed(range(n_tf)):
            root = rcm(i, root)
        root = pyr(root)
        return SRNet(
            x0_shape=x0_shape,
            y_shape=y_shape,
            root=root)
    return make_net

def dr_chain(type_, **hypers):
    def make_net():
        root = rcm(-1, reg())
        for i in reversed(range(len(arch) - 1)):
            root = rcm(i, reg(), root)
        root = pyr(root)
        return type_(
            x0_shape=x0_shape, y_shape=y_shape,
            root=root, **hypers)
    return make_net

def dr_tree(type_, **hypers):
    def make_net():
        root = pyr(
            rcm(0, reg(),
            rcm(1, reg(),
                rcm(2, reg(),
                rcm(3, reg(),
                    rcm(4, reg(),
                    rcm(5, reg())),
                    rcm(4, reg(),
                    rcm(5, reg())))),
                rcm(2, reg(),
                rcm(3, reg(),
                    rcm(4, reg(),
                    rcm(5, reg())),
                    rcm(4, reg(),
                    rcm(5, reg())))))))
        return type_(
            x0_shape=x0_shape, y_shape=y_shape,
            root=root, **hypers)
    return make_net

def ds_chain(**hypers):
    return dr_chain(DSNet, **hypers)

def ds_tree(**hypers):
    return dr_tree(DSNet, **hypers)

def cr_chain(**hypers):
    return dr_chain(CRNet, **hypers)

def cr_tree(**hypers):
    return dr_tree(CRNet, **hypers)
