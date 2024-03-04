import numpy as np
from cobyqa import minimize
from scipy.optimize import NonlinearConstraint, Bounds
import threading
import queue
from main import problem



x_queue = queue.Queue()
f_queue = queue.Queue()


def fun(x):
    x_queue.put(x)
    f_res = f_queue.get(timeout=30000.0)
    assert type(f_res)==float or f_res==np.nan or type(f_res) == np.float64, "f_res = "+str(f_res) + "| type = " + str(type(f_res))
    return f_res



class cobyqa:

    def __init__(self, problem: problem, seed):
        self.x_queue = x_queue
        self.f_queue = f_queue
        self.problem = problem
        self.rs = np.random.RandomState(seed+78)

        def minimz():
            while True:
                x0 = problem.random_feasible_sol(self.rs)
                minimize(fun, x0, bounds=bounds, constraints=constraints, options=options)
                print("Optimization end. Optimization algorithm restart.")

        bounds = Bounds([0.0 for _ in range(problem.dim)], [1.0 for _ in range(problem.dim)])
        constraints = NonlinearConstraint(lambda x: [el for el in problem.constraint_check(x)], 0.0, np.inf)
        options = {
        "disp": False,
        "radius_init": 0.1,
        "feasibility_tol": np.finfo(float).eps,
        }
        thread = threading.Thread(target=minimz)
        thread.start()


    def ask(self):
        return self.x_queue.get()

    def tell(self, f):
        self.f_queue.put(f)
 











