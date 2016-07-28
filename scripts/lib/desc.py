import numpy as np
import tensorflow as tf

__all__ = ['net_desc', 'render_net_desc']

################################################################################
# Network Statistics and Performance Metrics
################################################################################

def p_cor(net, ℓ):
    δ_cor = tf.equal(tf.argmax(ℓ.x, 1), tf.argmax(net.y, 1))
    return ℓ.p_ev * tf.to_float(δ_cor)

def p_inc(net, ℓ):
    δ_inc = tf.not_equal(tf.argmax(ℓ.x, 1), tf.argmax(net.y, 1))
    return ℓ.p_ev * tf.to_float(δ_inc)

def x_ent(net, ℓ, ϵ=1e-6):
    n_cls = net.y.get_shape()[1].value
    p_cls = ϵ / n_cls + (1 - ϵ) * ℓ.x
    return ℓ.p_ev * -tf.reduce_sum(net.y * tf.log(p_cls), 1)

def mse(net, ℓ):
    return ℓ.p_ev * tf.reduce_sum(tf.square(ℓ.x - net.y), 1)

def moc(net, ℓ):
    return ℓ.p_ev * ℓ.n_ops

def state_tensors(net):
    return {**{('p_cor', ℓ): p_cor(net, ℓ) for ℓ in net.leaves},
            **{('p_inc', ℓ): p_inc(net, ℓ) for ℓ in net.leaves},
            'acc': sum(p_cor(net, ℓ) for ℓ in net.leaves),
            'mxe': sum(x_ent(net, ℓ) for ℓ in net.leaves),
            'mse': sum(mse(net, ℓ) for ℓ in net.leaves),
            'moc': sum(moc(net, ℓ) for ℓ in net.layers)}

def mean_net_state(sess, net, batches):
    tensors = state_tensors(net)
    sums = {k: 0 for k in tensors.keys()}
    count = 0
    for x0, y in batches:
        samples = sess.run(tensors, {net.x0: x0, net.y: y})
        for k in tensors.keys():
            sums[k] += np.sum(samples[k], 0)
        count += len(x0)
    return {k: sums[k] / count for k in tensors.keys()}

################################################################################
# Descriptors
################################################################################

def layer_desc(ms_tr, ms_ts, ℓ):
    ℓ_is_leaf = len(ℓ.sinks) == 0
    return {'type': ℓ.__class__.__name__,
            'p_cor_tr': ms_tr['p_cor', ℓ] if ℓ_is_leaf else None,
            'p_cor_ts': ms_ts['p_cor', ℓ] if ℓ_is_leaf else None,
            'p_inc_tr': ms_tr['p_inc', ℓ] if ℓ_is_leaf else None,
            'p_inc_ts': ms_ts['p_inc', ℓ] if ℓ_is_leaf else None,
            'sinks': [layer_desc(ms_tr, ms_ts, s)
                      for s in ℓ.sinks]}

def net_desc(sess, net, dataset):
    ms_tr = mean_net_state(sess, net, dataset.training_batches())
    ms_ts = mean_net_state(sess, net, dataset.test_batches())
    return {'acc_tr': ms_tr['acc'], 'acc_ts': ms_ts['acc'],
            'mxe_tr': ms_tr['mxe'], 'mxe_ts': ms_ts['mxe'],
            'mse_tr': ms_tr['mse'], 'mse_ts': ms_ts['mse'],
            'moc_tr': ms_tr['moc'], 'moc_ts': ms_ts['moc'],
            'root': layer_desc(ms_tr, ms_ts, net.root)}

################################################################################
# Descriptor Rendering
################################################################################

def render_layer_desc(desc):
    annotation = (
        '[{:.1%} ✓ {:.1%} ×]'.format(desc['p_cor_ts'], desc['p_inc_ts'])
        if len(desc['sinks']) == 0 else '')
    sink_text = ''.join(
        '\n↳ ' + render_layer_desc(s).replace(
            '\n', '\n| ' if i < len(desc['sinks']) - 1 else '\n  ')
        for i, s in enumerate(desc['sinks']))
    return '%s %s%s' % (desc['type'], annotation, sink_text)

def render_net_desc(desc, name='Network'):
    return (
        '····························································\n'
        ' {}\n'
        '····························································\n'
        '⋮\n'
        '⋮   {}\n'
        '⋮\n'
        '⋮   Total accuracy: {:.2%}\n'
        '⋮   Mean cross-entropy: {:.2e}\n'
        '⋮   Mean squared error: {:.2e}\n'
        '⋮   Mean op count: {:.2e}\n'
        '⋮'
    ).format(
        name, render_layer_desc(desc['root']).replace('\n', '\n⋮   '),
        desc['acc_ts'], desc['mxe_ts'], desc['mse_ts'], desc['moc_ts'])
