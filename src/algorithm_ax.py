import numpy as np
import threading
import queue
from interfaces import *
from ax import optimize

# https://ax.dev/docs/bayesopt.html

class ax_optimizer:

    def __init__(self, prob:problem, seed: int, total_budget: int):
        self.prob = prob
        self.rs = np.random.RandomState(seed+78)
        self.total_budget = total_budget


        def fun(x):

            # numpy array for x from wierd framework format
            self.x_queue.put(np.array([el[1] for el in sorted(x.items(), key=lambda z: float(z[0].strip("x")))]))
            f_res = self.f_queue.get(timeout=30000.0)
            assert type(f_res)==float or type(f_res) == np.float64, "f_res = "+str(f_res) + "| type = " + str(type(f_res))
            return f_res

        self.x_queue = queue.Queue()
        self.f_queue = queue.Queue()

        def minimize():
            best_parameters, best_values, experiment, model = optimize(
                parameters=[
                {
                    "name": f"x{i}",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0.0, 1.0],
                } for i in range(prob.dim)
                ],
                evaluation_function=fun,
                minimize=True,
                total_trials=total_budget
            )

        thread = threading.Thread(target=minimize, args=[], daemon=True)
        thread.start()

        # print(best_parameters)
        # param.random_state = self.rs

    def ask(self):
        return self.x_queue.get()

    def tell(self, f):
        self.f_queue.put(f)

