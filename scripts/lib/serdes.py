import numpy as np
import tensorflow as tf

import lib.layer_types
import lib.net_types

__all__ = ['encode_net', 'decode_net', 'write_net', 'read_net']

################################################################################
# Layer Serialization/Deserialization
################################################################################

def encode_layer(layer):
    return None if layer is None else dict(
        type=type(layer).__name__, name=layer.name, hypers=vars(layer.hypers),
        params={k: v.eval() for k, v in vars(layer.params).items()},
        sinks=list(map(encode_layer, layer.sinks)),
        comps=list(map(encode_layer, layer.comps)),
        router=encode_layer(layer.router))

def decode_layer(record):
    return None if record is None else getattr(lib.layer_types, record['type'])(
        name=record['name'], router=decode_layer(record['router']),
        sinks=list(map(decode_layer, record['sinks'])),
        comps=list(map(decode_layer, record['comps'])),
        **{k: v for k, v in record['hypers'].items()})

def load_params(layer, record):
    return tf.no_op() if layer is None else tf.group(
        load_params(layer.router, record['router']),
        *(load_params(ℓ, r) for ℓ, r in zip(layer.comps, record['comps'])),
        *(load_params(ℓ, r) for ℓ, r in zip(layer.sinks, record['sinks'])),
        *(tf.assign(getattr(layer.params, k), v)
          for k, v in record['params'].items()))

################################################################################
# Network Serialization/Deserialization
################################################################################

def encode_net(net):
    return dict(
        type=type(net).__name__,
        root=encode_layer(net.root), hypers=vars(net.hypers),
        params={k: v.eval() for k, v in vars(net.params).items()})

def decode_net(record):
    type_ = getattr(lib.net_types, record['type'])
    root = decode_layer(record['root'])
    net = type_(root=root, **record['hypers'])
    load_params(net.root, record['root']).run()
    tf.group(*(
        tf.assign(getattr(net.params, k), v)
        for k, v in record['params'].items())).run()
    return net

def write_net(path, net):
    np.save(path, encode_net(net))

def read_net(path):
    return decode_net(np.load(path)[()])
