import tensorflow as tf

from lib.desc import net_desc, render_net_desc

################################################################################
# Network Training
################################################################################

def train(net, dataset, hypers=(lambda t: {}), batch_size=64,
          n_epochs=100, logging_period=5, name='Network'):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for t in range(n_epochs):
            for x0, y in dataset.training_batches(batch_size):
                net.train(x0, y, hypers(t))
            if (t + 1) % logging_period == 0:
                print(render_net_desc(
                    net_desc(sess, net, dataset),
                    '%s — Epoch %i' % (name, t + 1)))
        return net_desc(sess, net, dataset)
