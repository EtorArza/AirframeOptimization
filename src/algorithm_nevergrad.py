import numpy as np
import nevergrad as ng
from scipy.optimize import NonlinearConstraint, Bounds
import threading
import queue
from interfaces import *
import warnings



class ng_optimizer:

    def __init__(self, prob:problem, seed: int, parallel_threads: int, budget: int):
        self.prob = prob
        self.rs = np.random.RandomState(seed+78)
        x0 = self.prob.random_initial_sol()
        param = ng.p.Instrumentation(ng.p.Array(lower=0.0, upper=1.0, shape=x0.shape), seed=seed)
        param.random_state = self.rs
        self.budget = budget
        self.parallel_threads = parallel_threads
        self.optimizer = ng.optimizers.CMAbounded(parametrization=param, budget=budget, num_workers=parallel_threads)



    def ask(self):
        with warnings.catch_warnings(record=True) as wrngs:
            warnings.simplefilter("always")  # Set warning mode to always to catch warnings
            self.prev_sol = self.optimizer.ask()
            while len(wrngs) > 0:
                warning = wrngs.pop()
                if ' has already converged' in str(warning.message):
                    print(warning)
                    exit(1)
                        # if self.budget - self.n_f_evals > 10:
                        #     print(f"Reinitializing nevergrad with budget {self.budget - self.n_f_evals} left, as it already converged.")
                        #     x0 = self.prob.random_initial_sol()
                        #     param = ng.p.Instrumentation(ng.p.Array(lower=0.0, upper=1.0, init=x0))
                        #     self.optimizer = ng.optimizers.NGOpt(parametrization=param, budget=self.budget - self.n_f_evals, num_workers=self.parallel_threads,)
                else:
                    print(warning.message)

        self.prev_sol = self.optimizer.ask()

        return self.prev_sol[0][0].value

    def tell(self, f, x=None):
        assert x is None and self.optimizer.num_workers==1, "Parallelization has not been yet implemented."
        if self.prob.constraint_method == 'algo_specific':
            """
            Nevergrad uses the penalty method, where unfeasible solutions get penalized. The penalization is added to the loss function,
            and is computed by

            penalty = float(
                (1e5 + np.sum(np.maximum(f, 0.0))) * 
                sqrt(1 + self._num_tell) * 
                sqrt(np.sum(np.maximum(constraint_violation, 0.0)) # Choose largest violation, or 0 if all feasible.
            )

            Therefore, a positive value in one of the constraints means that the solution is unfeasible. This is the opposite to the standard we
            use in this codebase, where positive constraints means feasible (following nocedalNumericalOptimization2006 book). 
            """
            

            self.optimizer.tell(self.prev_sol, f, constraint_violation=-np.array(self.prob.constraint_check(self.prev_sol[0][0].value)))
        else:
            self.optimizer.tell(self.prev_sol, f)


