from skopt import gp_minimize
import scipy.optimize
from interfaces import *
import threading
import queue

class skoptbo_optimizer:

    def __init__(self, prob:problem, seed: int):
        self.prob = prob
        self.rs = np.random.RandomState(seed+78)
        def fun(x):
            self.x_queue.put(np.array(x))
            f_res = self.f_queue.get(timeout=30000.0)
            assert type(f_res)==float or type(f_res) == np.float64, "f_res = "+str(f_res) + "| type = " + str(type(f_res))
            return f_res

        self.fun = fun
        self.x_queue = queue.Queue()
        self.f_queue = queue.Queue()




        def minimize():
                gp_minimize(func=fun, 
                            dimensions=[(0.0,1.0)]*self.prob.dim, 
                            n_initial_points=100,
                            n_calls=99999,
                            random_state=self.rs,
                            noise=0.1,
                            n_jobs=8,
                            )

                
        thread = threading.Thread(target=minimize, args=[], daemon=True)
        thread.start()


    def ask(self):
        return self.x_queue.get()

    def tell(self, f):
        self.f_queue.put(f)
 
        
