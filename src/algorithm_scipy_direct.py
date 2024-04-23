import scipy
from scipy.optimize import NonlinearConstraint, Bounds
import scipy.optimize
from interfaces import *
import threading
import queue

class scipyDIRECT_optimizer:

    def __init__(self, prob:problem, seed: int):
        self.prob = prob
        self.rs = np.random.RandomState(seed+78)

        def fun(x):
            self.x_queue.put(np.array(x))
            f_res = self.f_queue.get(timeout=30000.0)
            assert type(f_res)==float or f_res==np.nan or type(f_res) == np.float64, "f_res = "+str(f_res) + "| type = " + str(type(f_res))
            return f_res

        self.fun = fun
        self.reinitialize()

    def reinitialize(self):
        bd = Bounds([0.0]*self.prob.dim,[1.0]*self.prob.dim)

        constraint_list = []
        if self.prob.constraint_method=="algo_specific":
            for i in range(self.prob.n_constraints):
                constraint_list.append(NonlinearConstraint(lambda x: self.prob.constraint_check(x)[i], 0.0, np.inf, keep_feasible=False))

        self.x_queue = queue.Queue()
        self.f_queue = queue.Queue()

        def minimize():
            scipy.optimize.direct(self.fun, bounds=bd, len_tol=0.0001, vol_tol=(0.01)**self.prob.dim,  locally_biased=False)
            print(f"Direct terminated after {self.prob.n_f_evals} evaluations. Exiting:")
            exit(1)

        thread = threading.Thread(target=minimize, args=[], daemon=True)
        thread.start()



    def ask(self):
        return self.x_queue.get()

    def tell(self, f):
        self.f_queue.put(f)
 
        
