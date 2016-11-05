from sys import maxsize

from lib.layer_types import MultiscaleConvMax

################################################################################
# Support Functions
################################################################################

def flatten(ℓ):
    return ([ℓ] if len(ℓ.comps) == 0
            else sum(map(flatten, ℓ.comps), []))

################################################################################
# Op Counting
################################################################################

def lazify_op_counts(ℓ, n_dep_ops=[]):
    for c in flatten(ℓ):
        if isinstance(c, MultiscaleConvMax):
            n_dep_ops = list(n_dep_ops)
            while len(n_dep_ops) < c.hypers.n_scales:
                n_dep_ops.append(0)
            for i, n in enumerate(reversed(c.n_ops_per_scale)):
                n_dep_ops[i] += n
            ℓ.n_ops -= c.n_ops
            ℓ.n_ops += n_dep_ops[0]
            n_dep_ops = n_dep_ops[1:]
    for s in ℓ.sinks:
        lazify_op_counts(s, n_dep_ops)
