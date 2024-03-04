import numpy as np
from numpy import cos, exp, pi
from pyOpt import NSGA2, SLSQP, Optimization
import threading
import queue
from main import problem

x_queue = queue.Queue()
f_queue = queue.Queue()

class pyopt:

    def __init__(self, problem: problem, seed):
        self.x_queue = x_queue
        self.f_queue = f_queue
        self.problem = problem
        self.seed = seed
        self.rs = np.random.RandomState(seed+78)

        def fun(x):
            self.x_queue.put(np.array(x))
            f_res = self.f_queue.get(timeout=30000.0)
            assert type(f_res)==float or f_res==np.nan or type(f_res) == np.float64, "f_res = "+str(f_res) + "| type = " + str(type(f_res))
            return f_res, [-el for el in problem.constraint_check(x)], 1 if type(f_res)==np.nan else 0


        # x0 = problem.random_feasible_sol(self.rs)

        opt_prob = Optimization(problem.problem_name, fun)
        opt_prob.addObj('fun')
        opt_prob.addVarGroup('x', problem.dim, 'c', lower=0.0, upper=1.0, )#value=x0)
        opt_prob.addConGroup('constraint', problem.n_constraints, 'i')

        def minimize(opt_prob):
            while True:
                nsga2 = NSGA2(options={'seed':self.rs.random()})
                nsga2(opt_prob)
                print("Refinment start.")
                slsqp = SLSQP()
                slsqp(opt_prob)
                raise NotImplementedError("Optimization terminated in child thread.")


        thread = threading.Thread(target=minimize, args=[opt_prob])
        thread.start()


    def ask(self):
        return self.x_queue.get()

    def tell(self, f):
        self.f_queue.put(f)
 







# def objfunc(x):
#     a = [3, 5, 2, 1, 7]
#     b = [5, 2, 1, 4, 9]
#     c = [1, 2, 5, 2, 3]
#     f = 0.0
#     for i in range(5):
#         f += -(c[i] * exp(-(1 / pi) * ((x[0] - a[i]) ** 2 + (x[1] - b[i]) ** 2)) * cos(
#             pi * ((x[0] - a[i]) ** 2 + (x[1] - b[i]) ** 2)))
#     g = [
#         20.04895 - (x[0] + 2.0) ** 2 - (x[1] + 1.0) ** 2,
#     ]
#     fail = 0
#     return f, g, fail
# opt_prob = Optimization('Langermann Function 11', objfunc)
# opt_prob.addVar('x1', 'c', lower=-2.0, upper=10.0, value=8.0)
# opt_prob.addVar('x2', 'c', lower=-2.0, upper=10.0, value=8.0)
# opt_prob.addObj('f')
# opt_prob.addCon('g', 'i')
# print(opt_prob)

# # Global Optimization
# nsga2 = NSGA2()
# nsga2(opt_prob)
# print(opt_prob.solution(0))

# # Local Optimization Refinement
# slsqp = SLSQP()
# slsqp(opt_prob.solution(0))
# print(opt_prob.solution(0).solution(0))






