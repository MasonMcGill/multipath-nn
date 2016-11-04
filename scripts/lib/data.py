import numpy as np
import numpy.random as rand
import scipy.io as io

__all__ = ['Dataset']

################################################################################
# Support Functions
################################################################################

def rand_flip(a):
    return a if rand.rand() < 0.5 else a[:, ::-1]

def rand_shift(a, r):
    b = np.empty_like(a)
    du, dv = rand.randint(-r, r + 1, 2)
    i_u_a = slice(max(du, 0), min(a.shape[0] + du, a.shape[0]))
    i_v_a = slice(max(dv, 0), min(a.shape[1] + dv, a.shape[1]))
    i_u_b = slice(max(-du, 0), min(a.shape[0] - du, a.shape[0]))
    i_v_b = slice(max(-dv, 0), min(a.shape[1] - dv, a.shape[1]))
    b[:] = np.mean(a, (0, 1))
    b[i_u_b, i_v_b] = a[i_u_a, i_v_a]
    return b

def augmented_batch(x0, y, n, r_shift):
    x0_batch = np.empty((n, *x0.shape[1:]))
    y_batch = np.empty((n, *y.shape[1:]))
    for i in range(n):
        j = rand.randint(0, len(x0))
        x0_batch[i] = rand_shift(rand_flip(x0[j]), r_shift)
        y_batch[i] = y[j]
    return x0_batch, y_batch

def batch(x0, y, n):
    i = rand.randint(0, len(x0), n)
    x0_batch = np.take(x0, i, axis=0)
    y_batch = np.take(y, i, axis=0)
    return x0_batch, y_batch

def full_set(x0, y, n):
    i = 0
    while i < len(x0):
        s = slice(i, min(i + n, len(x0)))
        yield x0[s], y[s]
        i += n

################################################################################
# Dataset
################################################################################

class Dataset:
    def __init__(self, path, n_vl=0):
        archive = io.loadmat(path)
        self.x0_tr = archive['x0_tr']
        self.x0_ts = archive['x0_ts']
        self.y_tr = archive['y_tr']
        self.y_ts = archive['y_ts']
        if n_vl > 0:
            rand.seed(0)
            order = rand.permutation(len(self.x0_tr))
            self.x0_vl = np.take(self.x0_tr, order[:n_vl], axis=0)
            self.y_vl = np.take(self.y_tr, order[:n_vl], axis=0)
            self.x0_tr = np.take(self.x0_tr, order[n_vl:], axis=0)
            self.y_tr = np.take(self.y_tr, order[n_vl:], axis=0)
            rand.seed(None)
        else:
            self.x0_vl = self.x0_tr[:0]
            self.y_vl = self.y_tr[:0]

    @property
    def x0_shape(self):
        return self.x0_tr.shape[1:]

    @property
    def y_shape(self):
        return self.y_tr.shape[1:]

    def augmented_training_batch(self, n=128, r_shift=4):
        return augmented_batch(self.x0_tr, self.y_tr, n, r_shift)

    def training_batch(self, n=128):
        return batch(self.x0_tr, self.y_tr, n)

    def validation_batch(self, n=128):
        return batch(self.x0_vl, self.y_vl, n)

    def test_batch(self, n=128):
        return batch(self.x0_ts, self.y_ts, n)

    def training_set(self, n=128):
        yield from full_set(self.x0_tr, self.y_tr, n)

    def validation_set(self, n=128):
        yield from full_set(self.x0_vl, self.y_vl, n)

    def test_set(self, n=128):
        yield from full_set(self.x0_ts, self.y_ts, n)
