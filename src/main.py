import sys
from tqdm import tqdm as tqdm
import numpy as np
import plot_src
from interfaces import *


if __name__ == "__main__":

    if sys.argv[1] == "--venn-diagram":
        assert sys.argv[2] in problem_name_list
        problem_name = sys.argv[2]
        n_montecarlo = 400
        prob = problem(problem_name)
        set_list = [set() for _ in range(prob.n_constraints)]
        for i in tqdm(range(n_montecarlo)):
            x_random = np.random.random(prob.dim)
            res = prob.constraint_check(x_random)
            [set_list[idx].add(i) for idx in range(prob.n_constraints) if res[idx]]
        plot_src.plot_venn_diagram(set_list, n_montecarlo, ["constraint_"+str(i) for i in range(prob.n_constraints)])


    elif sys.argv[1] == "--local-solve":
        problem_name = "toy"
        algorithm_name = "snobfit"
        seed = 4
        np.random.seed(seed)

        prob = problem(problem_name)
        algo = optimization_algorithm(prob, algorithm_name, seed)

        f_best = 1e10
        x_best = None
        print_every = 0
        import time
        ref = time.time()
        while prob.n_f_evals < 1000:
            x = algo.ask()
            f = prob.f_nan_on_unfeasible(x)
            algo.tell(f)
            if f < f_best:
                f_best = f
                x_best = x
                print("---")
                print(x_best, f_best)
            if print_every == 0:
                print("n_constraint_checks = ",prob.n_constraint_checks, "| n_f_evals", prob.n_f_evals, " | ", time.time() - ref ,"seconds")
                print_every = 1000
            else:
                print_every-=1
            

    else:
        print("sys.argv[1]=",sys.argv[1],"not recognized.", sep=" ")      