import numpy as np
import tensorflow as tf

__all__ = ['net_desc', 'render_net_desc']

################################################################################
# Descriptors
################################################################################

def mean_net_state(net, tensors, data, hypers):
    if len(tensors) == 0:
        return {}
    else:
        sums = {k: 0 for k in tensors.keys()}
        count = 0
        for x0, y in data:
            samples = net.eval(tensors, x0, y, hypers)
            for k in tensors.keys():
                sums[k] += np.sum(samples[k], 0)
            count += len(x0)
        return {k: sums[k] / count for k in tensors.keys()}

def layer_desc(ℓ, stats_tr, stats_ts):
    return {'type': type(ℓ).__name__, 'hypers': ℓ.hypers,
            'stats_tr': {k: v for (t, k), v in stats_tr.items() if t == ℓ},
            'stats_ts': {k: v for (t, k), v in stats_ts.items() if t == ℓ},
            'sinks': [layer_desc(s, stats_tr, stats_ts) for s in ℓ.sinks]}

def net_desc(net, dataset, hypers, net_state={}, layer_states={}):
    full_state = {
        **{(net, k): v for k, v in net_state.items()},
        **{(ℓ, k): v for ℓ, s in layer_states.items() for k, v in s.items()}}
    stats_tr = mean_net_state(net, full_state, dataset.training_batches(), hypers)
    stats_ts = mean_net_state(net, full_state, dataset.test_batches(), hypers)
    return {
        'type': type(net).__name__, 'hypers': net.hypers,
        'stats_tr': {k: v for (t, k), v in stats_tr.items() if t == net},
        'stats_ts': {k: v for (t, k), v in stats_ts.items() if t == net},
        'root': layer_desc(net.root, stats_tr, stats_ts)}

################################################################################
# Descriptor Rendering
################################################################################

def render_stats(stats):
    return (
        '(%s)' % '; '.join('%s=%.3g' % i for i in sorted(stats.items()))
        if len(stats) > 0 else '')

def render_layer_desc(desc, stats_key):
    sink_text = ''.join(
        '\n↳ ' + render_layer_desc(s, stats_key).replace(
            '\n', '\n| ' if i < len(desc['sinks']) - 1 else '\n  ')
        for i, s in enumerate(desc['sinks']))
    return '%s %s%s' % (desc['type'], render_stats(desc[stats_key]), sink_text)

def render_net_desc(desc, name='Network'):
    return (
        '┌───────────────────────────────────────────────────────────\n'
        '│ {name}\n'
        '├───────────────────────────────────────────────────────────\n'
        '│ Training Set:\n'
        '│\n'
        '│   [{net_type}] {net_stats_tr}\n'
        '│     {layers_tr}\n'
        '│\n'
        '│ Test Set:\n'
        '│\n'
        '│   [{net_type}] {net_stats_ts}\n'
        '│     {layers_ts}\n'
        '│').format(
            name=name,
            net_type=desc['type'],
            net_stats_tr=render_stats(desc['stats_tr']),
            net_stats_ts=render_stats(desc['stats_ts']),
            layers_tr=render_layer_desc(desc['root'], 'stats_tr')
                      .replace('\n', '\n│     '),
            layers_ts=render_layer_desc(desc['root'], 'stats_ts')
                      .replace('\n', '\n│     '))
