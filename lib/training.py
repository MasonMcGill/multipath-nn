import numpy as np
import tensorflow as tf

def mean_over(batches, sess, net, func):
    sums = np.zeros(len(list(net.layers)), object)
    count = 0
    for x0, y in batches:
        stats = sess.run(list(map(func, net.layers)), {net.x0: x0, net.y: y})
        for i in range(len(sums)):
            sums[i] += np.sum(stats[i], 0)
        count += len(x0)
    return {l: s / count for l, s in zip(net.layers, sums)}

def describe(layer, λ_cor, λ_inc):
    if len(layer.sinks) == 0:
        return (
            layer.__class__.__name__
            + ' [%.1f%% ✓ %.1f%% ×]'
            % (100 * λ_cor[layer], 100 * λ_inc[layer]))
    else:
        return (
            layer.__class__.__name__
            + ''.join(
                '\n↳ ' + describe(s, λ_cor, λ_inc).replace(
                    '\n', '\n| ' if i < len(layer.sinks) - 1 else '\n  ')
                for i, s in enumerate(layer.sinks)))

def log_progress(sess, net, dataset, t, batch_size=512):
    def p_cor(layer):
        if len(layer.sinks) == 0:
            δ_cor = tf.equal(tf.argmax(layer.x, 1), tf.argmax(net.y, 1))
            return layer.p_ev * tf.to_float(δ_cor)
        else:
            return tf.zeros(tf.shape(layer.x)[:1])
    def p_inc(layer):
        if len(layer.sinks) == 0:
            δ_inc = tf.not_equal(tf.argmax(layer.x, 1), tf.argmax(net.y, 1))
            return layer.p_ev * tf.to_float(δ_inc)
        else:
            return tf.zeros(tf.shape(layer.x)[:1])
    λ_cor = mean_over(dataset.test_batches(batch_size), sess, net, p_cor)
    λ_inc = mean_over(dataset.test_batches(batch_size), sess, net, p_inc)
    print('····························································')
    print(' Epoch %i' % t)
    print('····························································')
    print(('⋮\n%s\n' % describe(net.root, λ_cor, λ_inc))
              .replace('\n', '\n⋮   '))

def train(net, dataset, batch_size=64, n_epochs=100, logging_period=5):
    train_op = tf.train.AdamOptimizer().minimize(tf.reduce_mean(net.ℓ_tr))
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for t in range(n_epochs):
            if t % logging_period == 0:
                log_progress(sess, net, dataset, t)
            for x0, y in dataset.training_batches(batch_size):
                train_op.run({net.x0: x0, net.y: y})
        log_progress(sess, net, n_epochs)
