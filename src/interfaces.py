import numpy as np


problem_name_list = ["airframes", "windflo", "toy"]
algorithm_name_list = ["snobfit", "cobyqa", "pyopt"]

class problem:

    def __init__(self, problem_name):
        assert problem_name in problem_name_list
        self.problem_name = problem_name

        self.n_f_evals = 0
        self.n_constraint_checks = 0

        if problem_name == "airframes":
            import problem_airframes
            self.dim = 15
            self._constraint_check = problem_airframes.constraint_check_hexarotor_0_1
            self._f = lambda x: problem_airframes.f_symmetric_hexarotor_0_1(x, target=[2.3,0.75,1.5])[0]
            self.plot_solution = lambda x: problem_airframes.plot_airframe_design(problem_airframes._decode_symmetric_hexarotor_to_RobotParameter(x))

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
        assert type(x) == np.ndarray
        self.n_f_evals+=1
        return self._f(x)
    
    def constraint_check(self, x):
        assert type(x) == np.ndarray
        self.n_constraint_checks+=1
        return self._constraint_check(x)

    def plot_solution(self, x):
        pass
    
    def f_nan_on_unfeasible(self, x):
        feasible = np.min(self.constraint_check(x)) > 0
        return self.f(x) if feasible else np.nan
    
    def random_feasible_sol(self, np_random_state):
        feasible = False
        while not feasible:
            x0 = np_random_state.random(self.dim)
            feasible = np.min(self.constraint_check(x0)) > 0
        return x0

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
        elif algorithm_name == "cobyqa":
            import algorithm_cobyqa
            self.algo = algorithm_cobyqa.cobyqa(problem, seed)
        elif algorithm_name == "pyopt":
            import algorithm_pyopt
            self.algo = algorithm_pyopt.pyopt(problem, seed)
        else:
            print("Algorithm name", algorithm_name, "not recognized.")

    def ask(self):
        return self.algo.ask()

    def tell(self, f):
        self.algo.tell(f)
