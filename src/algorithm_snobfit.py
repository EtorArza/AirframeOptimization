"""
https://github.com/scikit-quant/scikit-quant
https://arnold-neumaier.at/software/snobfit/
https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.optimizers.SNOBFIT.html # <- the most useful link

"""


import numpy as np
from skquant.opt import minimize
from main import problem
import threading
import queue


class snobfit:

    def __init__(self, problem: problem, seed):
        self.problem = problem
        self.rs = np.random.RandomState(seed+78)
        x0 = self.rs.random(problem.dim)
        self.x_queue = queue.Queue()
        self.f_queue = queue.Queue()
        bounds = np.array([[0, 1] for _ in range(problem.dim)], dtype=float)
        thread = threading.Thread(target=minimize, args=(self.x_queue, self.f_queue, lambda x: ValueError("This parameter should not be used."), x0, bounds), kwargs={'budget':1000000, 'method':'SnobFit', 'maxfail':np.inf})
        thread.start()

    def ask(self):
        return self.x_queue.get()

    def tell(self, f):
        self.f_queue.put(f)





