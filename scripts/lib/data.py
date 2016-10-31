from os import listdir
from os.path import join
from queue import Queue
from threading import Thread
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

class Dataset:
    def __init__(self, path):
        self.path = path
        self.n_pts = {
            mode: len(listdir(join(self.path, mode, 'x0')))
            for mode in ['tr', 'vl', 'ts']}
        self.task_queue = Queue()
        def exec_read_loop():
            γ_dec = np.float32((np.arange(256) / 255)**2.2)
            while True:
                task = self.task_queue.get()
                task.x0 = np.array([
                    cv2.LUT(cv2.imread(task.x0_fmt % j), γ_dec)
                    for j in task.indices])
                task.y = np.array([
                    np.load(task.y_fmt % j)
                    for j in task.indices])
                self.task_queue.task_done()
        Thread(target=exec_read_loop).start()

    @property
    def x0_shape(self):
        x0_path = join(self.path, 'tr/x0/%.8i.jpg' % 0)
        return cv2.imread(x0_path).shape

    @property
    def y_shape(self):
        y_path = join(self.path, 'tr/y/%.8i.npy' % 0)
        return np.load(y_path).shape

    def _batches(self, mode, n_batches, batch_size):
        n_pts = self.n_pts[mode]
        task = Ns(x0_fmt=join(self.path, mode, 'x0', '%.8i.jpg'),
                  y_fmt=join(self.path, mode, 'y', '%.8i.npy'),
                  indices=rand.randint(0, n_pts, batch_size))
        self.task_queue.put(task)
        for _ in range(n_batches):
            self.task_queue.join()
            x0, y = task.x0, task.y
            task.indices = rand.randint(0, n_pts, batch_size)
            self.task_queue.put(task)
            yield x0, y

    def _full_set(self, mode, batch_size):
        n_pts = self.n_pts[mode]
        task = Ns(x0_fmt=join(self.path, mode, 'x0', '%.8i.jpg'),
                  y_fmt=join(self.path, mode, 'y', '%.8i.npy'),
                  indices=range(batch_size))
        self.task_queue.put(task)
        for i in range(0, n_pts, batch_size):
            self.task_queue.join()
            x0, y = task.x0, task.y
            task.indices = range(i, i + batch_size)
            self.task_queue.put(task)
            yield x0, y

    def training_batches(self, n_batches, batch_size=128):
        yield from self._batches('tr', n_batches, batch_size)

    def validation_batches(self, n_batches, batch_size=128):
        yield from self._batches('vl', n_batches, batch_size)

    def test_batches(self, n_batches, batch_size=128):
        rand.seed(0)
        yield from self._batches('ts', n_batches, batch_size)
        rand.seed()

    def full_trainin_set(self, batch_size=128):
        yield from self._full_set('tr', batch_size)

    def full_validation_set(self, batch_size=128):
        yield from self._full_set('vl', batch_size)

    def full_test_set(self, batch_size=128):
        yield from self._full_set('ts', batch_size)
