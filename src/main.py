import sys
from tqdm import tqdm as tqdm
import numpy as np
import plot_src


np.random.seed(2)

problem_name_list = ["airframes"]
class problem:

    def __init__(self, problem_name):
        assert problem_name in problem_name_list
        self.problem_name = problem_name

        if problem_name == "airframes":
            import problem_airframes
            self.dim = 6*4
            self.constraint_check = problem_airframes.constraint_check
            self.plot_solution = problem_airframes.plot_airframe_design

        self.n_constraints = len(self.constraint_check(np.random.random(self.dim)))

    def f(self, x):
        pass
    
    def constraint_check(self, x):
        pass

    def plot_solution(self, x):
        pass


class encoding:

    def __init__(self, encoding_name, dim):
        self.encoding_name = encoding_name
        self.dim = dim


class optimization_algorithm:

    def __init__(self, problem, algorithm_name):
        self.problem = problem
        pass

    def ask(self):
        pass

    def tell(self, f):
        pass


if __name__ == "__main__":
    if sys.argv[1] == "--venn-diagram":
        n_montecarlo = 400
        for problem_name in problem_name_list:
            prob = problem(problem_name)
            set_list = [set() for _ in range(prob.n_constraints)]
            for i in tqdm(range(n_montecarlo)):
                x_random = np.random.random(prob.dim)
                res = prob.constraint_check(x_random)
                [set_list[idx].add(i) for idx in range(prob.n_constraints) if res[idx]]
            plot_src.plot_venn_diagram(set_list, n_montecarlo, ["constraint_1", "constraint_2"])
