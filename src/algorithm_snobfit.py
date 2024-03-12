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
        self.x_queue = queue.Queue()
        self.f_queue = queue.Queue()
        self.problem = problem
        self.rs = np.random.RandomState(seed+78)
        def minimiz():
            while True:
                try:
                    x0 = self.rs.random(problem.dim)
                    bounds = np.array([[0, 1] for _ in range(problem.dim)], dtype=float)
                    minimize(self.x_queue, self.f_queue, lambda x: ValueError("This parameter should not be used."), x0, bounds, budget=100000, method='SnobFit', maxfail=np.inf)
                except Exception as e:
                    print("Error on Snobfit:", e, "child thread crashed.")
                    exit(1)
                print("Optimization end. Optimization algorithm restart.")
        

        thread = threading.Thread(target=minimiz, args=(), kwargs={}, daemon=True)
        thread.start()

    def ask(self):
        return self.x_queue.get()

    def tell(self, f):
        self.f_queue.put(f)





