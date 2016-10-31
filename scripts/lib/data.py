from multiprocessing import Array, Lock, Process, Value
from os import listdir
from os.path import join
from time import sleep
from types import SimpleNamespace as Ns

import cv2
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
# Datasets
################################################################################

def sample_buf(n, x0_shape, y_shape):
    x0_mem = Array('f', n * int(np.prod(x0_shape)), lock=False)
    y_mem = Array('f', n * int(np.prod(y_shape)), lock=False)
    return Ns(x0=np.reshape(np.frombuffer(x0_mem, 'f'), (n, *x0_shape)),
              y=np.reshape(np.frombuffer(y_mem, 'f'), (n, *y_shape)),
              i_start=Value('i', 0, lock=False),
              n_loaded=Value('i', 0, lock=False),
              n_loading=Value('i', 0, lock=False),
              lock=Lock())

class Dataset:
    def __init__(self, path):
        self.path = path
        self.x0_shape = cv2.imread(join(self.path, 'tr/x0/%.8i.jpg' % 0)).shape
        self.y_shape = np.load(join(self.path, 'tr/y/%.8i.npy' % 0)).shape
        self.n_pts = {
            mode: len(listdir(join(self.path, mode, 'x0')))
            for mode in ['tr', 'vl', 'ts']}
        self.buf = {
            mode: sample_buf(512, self.x0_shape, self.y_shape)
            for mode in ['tr', 'vl', 'ts']}
        def exec_read_loop():
            γ_dec = np.float32((np.arange(256) / 255)**2.2)
            while True:
                full = True
                while full:
                    for mode, buf in self.buf.items():
                        with buf.lock:
                            n_fresh = buf.n_loaded.value + buf.n_loading.value
                            n_slots = len(buf.x0)
                            if n_fresh < n_slots:
                                dst_i = (buf.i_start.value + n_fresh) % n_slots
                                buf.n_loading.value += 1
                                full = False
                                break
                    if full:
                        sleep(0.001)
                src_i = rand.randint(0, self.n_pts[mode])
                x0_path = join(self.path, mode, 'x0', '%.8i.jpg' % src_i)
                y_path = join(self.path, mode, 'y', '%.8i.npy' % src_i)
                buf.x0[dst_i] = cv2.LUT(cv2.imread(x0_path), γ_dec)
                buf.y[dst_i] = np.load(y_path)
                with buf.lock:
                    buf.n_loading.value -= 1
                    buf.n_loaded.value += 1
        for _ in range(4):
            Process(target=exec_read_loop).start()

    def _batches(self, mode, n_batches, batch_size):
        buf = self.buf[mode]
        for _ in range(n_batches):
            while True:
                with buf.lock:
                    n_fresh = buf.n_loaded.value + buf.n_loading.value
                    if buf.n_loaded.value >= batch_size:
                        x0_i = buf.x0[buf.i_start.value:][:batch_size]
                        y_i = buf.y[buf.i_start.value:][:batch_size]
                        buf.n_loaded.value -= batch_size
                        buf.i_start.value += batch_size
                        buf.i_start.value %= len(buf.x0)
                        break
                sleep(0.001)
            yield x0_i, y_i

    def _full_set(self, mode, batch_size):
        pass

    def training_batches(self, n_batches, batch_size=128):
        yield from self._batches('tr', n_batches, batch_size)

    def validation_batches(self, n_batches, batch_size=128):
        yield from self._batches('vl', n_batches, batch_size)

    def test_batches(self, n_batches, batch_size=128):
        yield from self._batches('ts', n_batches, batch_size)

    def full_trainin_set(self, batch_size=128):
        yield from self._full_set('tr', batch_size)

    def full_validation_set(self, batch_size=128):
        yield from self._full_set('vl', batch_size)

    def full_test_set(self, batch_size=128):
        yield from self._full_set('ts', batch_size)
