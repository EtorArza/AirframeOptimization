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
import time
from tqdm import tqdm as tqdm

x_queue = queue.Queue()
f_queue = queue.Queue()


class snobfit:

    def __init__(self, problem: problem):
        self.problem = problem
        x0 = np.random.random(problem.dim)
        # self.minimizer = minimize(paused_flag, None, x0, bounds, np.inf, method='SnobFit', maxfail=np.inf)

    def show(self):
        pass

    def tell(self):
        pass


# some interesting objective function to minimize
def objective_function(x):
    if x[0] < 0.8:
        return np.nan
    fv = np.inner(x, x)
    fv *= 1 + 0.1*np.sin(10*(x[0]+x[1]+x[2]))
    return np.random.normal(fv, 0.01)

# create a numpy array of bounds, one (low, high) for each parameter
bounds = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=float)



# initial values for all parameters
x0 = np.array([0.5, 0.5, 0.5])





thread = threading.Thread(target=minimize, args=(x_queue, f_queue, objective_function, x0, bounds), kwargs={'budget':1000000, 'method':'SnobFit', 'maxfail':np.inf})
thread.start()

for i in tqdm(range(10000)):
    x = x_queue.get()
    f = objective_function(x)
    f_queue.put(f)
    print(x, f)




