import numpy as np
import numpy.typing
import os
from cobyqa import minimize
from scipy.optimize import NonlinearConstraint, Bounds


def f(x: numpy.typing.ArrayLike):
    # If solution not feasible, return np.nan
    if not np.all(constraint_check(x)):
        return np.nan
    return np.linalg.norm(x)

def constraint_check(x: numpy.typing.ArrayLike):
    # constraint 0:  > x[2] > x[3]
    check_0 = x[0] - x[1]
    check_1 = x[1] - x[2]
    check_2 = x[2] - x[3]
    check_3 = x[3] - x[4]

    return (check_0, check_1, check_2, check_3) 

def random_feasible_sol(dim, ):
    feasible = False
    seed = 2
    rs = np.random.RandomState(seed)
    while not feasible:
        x0 = rs.random(dim)
        feasible = constraint_check(x0)
    return x0

problem_dim = 50
x0 = random_feasible_sol(problem_dim) # get a random feasible solution to initialize algorithm.
print(x0)
bounds = Bounds([0.0 for _ in range(problem_dim)], [1.0 for _ in range(problem_dim)])
constraints = NonlinearConstraint(lambda x: [constraint_check(x)], 0.0, np.inf)

options = {
"disp": True,
"feasibility_tol": np.finfo(float).eps,
}

minimize(f, x0, bounds=bounds, constraints=constraints, options=options)
