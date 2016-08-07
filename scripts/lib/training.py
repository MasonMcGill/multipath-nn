import tensorflow as tf

from lib.desc import net_desc, render_net_desc

################################################################################
# Network Training
################################################################################

def train(net, dataset, name='Network', batch_size=64,
          n_epochs=100, logging_period=5):
    net_t = tf.Variable(0.0)
    λ_learn = 0.1 / 2**(net_t / 10)
    optimizer = tf.train.MomentumOptimizer(λ_learn, 0.9)
    train_op = optimizer.minimize(tf.reduce_mean(net.ℓ_tr))
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for t in range(n_epochs):
            for x0, y in dataset.training_batches(batch_size):
                train_op.run({net.x0: x0, net.y: y, net_t: t})
            if (t + 1) % logging_period == 0:
                desc = net_desc(sess, net, dataset)
                print(render_net_desc(desc, '%s — Epoch %i' % (name, t + 1)))
        return net_desc(sess, net, dataset)
