import numpy as np


problem_name_list = ["airframes", "windflo", "toy"]
algorithm_name_list = ["snobfit", "cobyqa", "pyopt"]
constraint_method_list = ['ignore','nan_on_unfeasible','constant_penalty_no_evaluation','algo_specific']

class problem:

    def __init__(self, problem_name, constraint_method):
        '''
        Initializes the problem to be solved.
        
        Parameters:
            problem_name (str): The name of the problem.
            constraint_method (str): How to deal with constraints. Supported values are 'ignore','nan_on_unfeasible', 'constant_penalty_no_evaluation', 'algo_specific'.
        '''
        assert problem_name in problem_name_list
        assert constraint_method in constraint_method_list
        self.constraint_method = constraint_method
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

        return_value = None
        if self.constraint_method == 'ignore' or self.constraint_method =='algo_specific':
            return_value = 'f'
        elif self.constraint_method == 'nan_on_unfeasible':
            return_value = 'f' if np.all(np.array(self.constraint_check(x)) > 0) else np.nan
        elif self.constraint_method == 'constant_penalty_no_evaluation':
            return_value = 'f' if np.all(np.array(self.constraint_check(x)) > 0) else np.inf
        else:
            raise ValueError("Constraint method "+str(self.constraint_method)+" not recognized.")

        if return_value=='f':
            return_value = self._f(x)
            self.n_f_evals+=1
        return return_value
    
    def constraint_check(self, x) -> tuple:
        assert type(x) == np.ndarray
        self.n_constraint_checks+=1
        res = self._constraint_check(x)
        assert type(res) == tuple, str(res) + " of type " + str(type(res))
        return res

    def plot_solution(self, x):
        pass

    def random_initial_sol(self, np_random_state):

        if self.constraint_method == 'ignore' or self.constraint_method =='algo_specific':
            return np_random_state.random(self.dim)
        elif self.constraint_method == 'nan_on_unfeasible' or 'constant_penalty_no_evaluation':
            feasible = False
            while not feasible:
                x0 = np_random_state.random(self.dim)
                feasible = np.all(np.array(self.constraint_check(x0)) > 0)
            return x0
        else:
            raise ValueError("Constraint method "+str(self.constraint_method)+" not recognized.")


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