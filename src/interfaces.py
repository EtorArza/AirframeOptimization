import numpy as np


problem_name_list = ["airframes", "windflo", "toy"]
algorithm_name_list = ["snobfit"]

class problem:

    def __init__(self, problem_name):
        assert problem_name in problem_name_list
        self.problem_name = problem_name

        self.n_f_evals = 0
        self.n_constraint_checks = 0

        if problem_name == "airframes":
            import problem_airframes
            self.dim = 6*4
            self._constraint_check = problem_airframes.constraint_check
            self.plot_solution = problem_airframes.plot_airframe_design

        if problem_name == "windflo":
            import problem_windflo
            self.dim = problem_windflo.SOLUTION_DIM
            self._constraint_check = problem_windflo.constraint_check
            self._f = problem_windflo.f
            self.plot_solution = problem_windflo.plot_WindFLO

        if problem_name == "toy":
            dim = 50
            import problem_toy
            self.dim = dim
            self._constraint_check = problem_toy.constraint_check
            self._f = problem_toy.f

        self.n_constraints = len(self._constraint_check(np.random.random(self.dim)))

    def f(self, x):
        self.n_f_evals+=1
        return self._f(x)
    
    def constraint_check(self, x):
        self.n_constraint_checks+=1
        return self._constraint_check(x)

    def plot_solution(self, x):
        pass
    
    def f_nan_on_unfeasible(self, x):
        feasible = np.all(self.constraint_check(x))
        return self.f(x) if feasible else np.nan

class encoding:

    def __init__(self, encoding_name, dim):
        self.encoding_name = encoding_name
        self.dim = dim


class optimization_algorithm:

    def __init__(self, problem, algorithm_name, seed):
        assert algorithm_name in algorithm_name_list
        self.problem = problem
        if algorithm_name == "snobfit":
            import algorithm_snobfit
            self.algo = algorithm_snobfit.snobfit(problem, seed)
        pass

    def ask(self):
        return self.algo.ask()

    def tell(self, f):
        self.algo.tell(f)
