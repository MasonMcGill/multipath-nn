import numpy as np
import numpy.random as rand
import scipy.io as io

################################################################################
# Support Functions
################################################################################

def batches(x0, y, n):
    order = rand.permutation(len(x0))
    x0_shuf = np.take(x0, order, axis=0)
    y_shuf = np.take(y, order, axis=0)
    for i in range(0, len(x0) - n + 1, n):
        yield x0_shuf[i:i+n], y_shuf[i:i+n]

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

    def training_batches(self, n=128):
        yield from batches(self.x0_tr, self.y_tr, n)

    def validation_batches(self, n=128):
        yield from batches(self.x0_vl, self.y_vl, n)

    def test_batches(self, n=128):
        yield from batches(self.x0_ts, self.y_ts, n)
