import numpy as np
import nevergrad as ng
from scipy.optimize import NonlinearConstraint, Bounds
import threading
import queue
from interfaces import *




class ng_optimizer:

    def __init__(self, prob:problem, seed: int, parallel_threads: int, budget: int):
        self.prob = prob
        self.rs = np.random.RandomState(seed+78)
        x0 = self.prob.random_initial_sol(self.rs)
        param = ng.p.Instrumentation(ng.p.Array(lower=0.0, upper=1.0, init=x0))
        self.optimizer = ng.optimizers.NGOpt(parametrization=param, budget=budget, num_workers=parallel_threads, )

        if self.prob.constraint_method == 'algo_specific':
            # Repeatedly samples points without evaluating, such that only feasible points are sampled.
            def bool_constraint(x):
                assert len(x[0])==1, x
                return np.all(np.array(self.prob.constraint_check(x[0][0])) > 0)

            self.optimizer.parametrization.register_cheap_constraint(bool_constraint)
        else:
            pass

    def ask(self):
        self.prev_sol = self.optimizer.ask()
        return self.prev_sol[0][0].value

    def tell(self, f, x=None):
        assert x is None and self.optimizer.num_workers==1, "Parallelization has not been yet implemented."
        self.optimizer.tell(self.prev_sol, f)


