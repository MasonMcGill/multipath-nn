from random import shuffle

import tensorflow as tf

from lib.desc import net_desc, render_net_desc

################################################################################
# Network Statistics
################################################################################

def p_cor(net, ℓ):
    δ_cor = tf.equal(tf.argmax(ℓ.x, 1), tf.argmax(net.y, 1))
    return ℓ.p_ev * tf.to_float(δ_cor)

def p_inc(net, ℓ):
    δ_inc = tf.not_equal(tf.argmax(ℓ.x, 1), tf.argmax(net.y, 1))
    return ℓ.p_ev * tf.to_float(δ_inc)

################################################################################
# Network Training
################################################################################

def train(net, dataset, validate=True, hypers=(lambda t: {}), batch_size=256,
          n_epochs=100, logging_period=5, name='Network'):
    net_state = {
        (net, 'acc'): sum(p_cor(net, ℓ) for ℓ in net.leaves),
        (net, 'moc'): sum(ℓ.p_ev * ℓ.n_ops for ℓ in net.layers),
        **{(ℓ, 'p_cor'): p_cor(net, ℓ) for ℓ in net.leaves},
        **{(ℓ, 'p_inc'): p_inc(net, ℓ) for ℓ in net.leaves}}
    for t in range(n_epochs):
        ϕ = hypers(t)
        set_tr = [('tr', b) for b in dataset.training_batches(batch_size)]
        set_vl = [('vl', b) for b in dataset.validation_batches(batch_size)]
        batches = set_tr + set_vl
        shuffle(batches)
        for mode, (x0, y) in batches:
            if validate and mode == 'vl':
                net.validate(x0, y, ϕ)
            else:
                net.train(x0, y, ϕ)
        if (t + 1) % logging_period == 0:
            print(render_net_desc(
                net_desc(net, dataset, ϕ, net_state),
                '%s — Epoch %i' % (name, t + 1)))
    return net_desc(net, dataset, hypers(n_epochs), net_state)

################################################################################
# Network Training Profiling
################################################################################

def profile(path, net, dataset, hypers=(lambda t: {}),
            batch_size=256, n_warm=10):
    ϕ = hypers(0)
    run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_meta = tf.RunMetadata()
    for i, (x0, y) in enumerate(dataset.training_batches(batch_size)):
        feed = {net.x0: x0, net.y: y, net.mode: 'tr', **ϕ}
        if i < n_warm:
            net.sess.run(net.train_op, feed, run_opts)
        else:
            net.sess.run(net.train_op, feed, run_opts, run_meta)
            tl = tf.python.client.timeline.Timeline(run_meta.step_stats)
            with open(path, 'w') as f:
                f.write(tl.generate_chrome_trace_format())
            break
