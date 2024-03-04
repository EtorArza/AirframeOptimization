import sys
from tqdm import tqdm as tqdm
import numpy as np
from interfaces import *


if __name__ == "__main__":

    sys.argv.append("--local-solve")

    # Show the percentage of feasible solutions in a problem
    if sys.argv[1] == "--venn-diagram":
        assert sys.argv[2] in problem_name_list
        import plot_src
        problem_name = sys.argv[2]
        n_montecarlo = 1200
        prob = problem(problem_name)
        set_list = [set() for _ in range(prob.n_constraints)]
        for i in tqdm(range(n_montecarlo)):
            x_random = np.random.random(prob.dim)
            res = prob.constraint_check(x_random)
            [set_list[idx].add(i) for idx in range(prob.n_constraints) if res[idx] > 0.0]
        plot_src.plot_venn_diagram(set_list, n_montecarlo, ["constraint_"+str(i) for i in range(prob.n_constraints)])

    # Directly solve problem locally, with f function that returns np.nan on infeasible solutions.
    elif sys.argv[1] == "--local-solve":
        problem_name = "toy"
        algorithm_name = "pyopt"
        verbose = False
        seed = 4
        np.random.seed(seed)

        prob = problem(problem_name)
        algo = optimization_algorithm(prob, algorithm_name, seed)

        f_best = 1e10
        x_best = None
        print_every = 0
        import time
        ref = time.time()
        while prob.n_f_evals < 1000000:
            x = algo.ask()
            f = prob.f(x)
            if verbose:
                print("n_f_evals:", prob.n_f_evals, "n_constraint_checks:", prob.n_constraint_checks, "x:", x, "f:", f)
            algo.tell(f)
            if f < f_best:
                f_best = f
                x_best = x
                print("---")
                print(x_best, f_best)
            if print_every == 0:
                print("n_constraint_checks = ",prob.n_constraint_checks, "| n_f_evals", prob.n_f_evals, " | ", time.time() - ref ,"seconds")
                print_every = 1000
                ref = time.time()
            else:
                print_every-=1

        print("-----------")
        print("-----------")
        print("Finished local optimization.")
        print("n_f_evals:", prob.n_f_evals, "n_constraint_checks:", prob.n_constraint_checks, "x:", x, "f:", f)
        print("-----------")
        print("-----------")
        exit(0)

    # Plot how time per evaluation in snobfit increases linearly
    elif sys.argv[1] == "--plot-snobfit-time-per-1000-evaluations":
        from matplotlib import pyplot as plt
        a = [35.672,66.487,83.07,106.848,138.234,164.302,165.799,150.121,209.833,222.891,271.217,238.373,299.406,275.698,383.726,350.589,306.331,362.808,344.787,466.726000000001,495.216,544.442,483.626,572.485,505.259,523.818,644.181000000001,629.82,687.755999999999,601.378000000001,670.198,619.061,793.431000000001,791.849,738.907999999999,605.154999999999,705.071000000002,752.395999999999,727.92,760.445,774.154000000002,1010.754,1039.556,1055.197,1093.278,1070.267,1209.765,1240.372,1382.626,1265.622,1418.167,1332.901,1192.581,1425.083,1391.77,1585.827,1363.194,1522.193,1248.111,1561.504,1648.914,1860.214,1868.236,1732.594,1893.023,1554.103,2117.976,1658.75999999999,1469.963,1273.931]
        plt.plot(a)
        plt.ylabel("time (s)")
        plt.xlabel("x 1000 evaluations")
        plt.title("Snobfit: time per 1000 evaluations")
        plt.show()

    else:
        print("sys.argv[1]=",sys.argv[1],"not recognized.", sep=" ")
