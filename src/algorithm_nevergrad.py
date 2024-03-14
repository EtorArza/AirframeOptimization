import numpy as np
import nevergrad as ng
from scipy.optimize import NonlinearConstraint, Bounds
import threading
import queue
from main import problem





class ng_optimizer:

    def __init__(self, prob: problem, seed: int, parallel_threads: int, budget: int):
        self.prob = problem
        self.rs = np.random.RandomState(seed+78)
        x0 = self.prob.random_initial_sol(self.rs)
        instrum = ng.p.Instrumentation(ng.p.Array(shape=(self.prob.dim,), lower=0.0, upper=1.0, init=x0))
        self.optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=100, num_workers=parallel_threads)

        if self.prob.constraint_method == 'algo_specific':
            # Repeatedly samples points without evaluating, such that only feasible points are sampled.
            self.optimizer.parametrization.register_cheap_constraint(lambda x: np.array(self.prob.constraint_check(x)))
        else:
            pass

    def ask(self):
        self.prev_sol = self.optimizer.ask()
        return self.prev_sol

    def tell(self, f, x=None):
        assert not x is None or self.optimizer.num_workers==1
        self.optimizer.tell(self.prev_sol, f)


