import sys
from tqdm import tqdm as tqdm
import numpy as np
from interfaces import *


if __name__ == "__main__":

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
        sys.argv.pop()
        problem_name = "airframes"
        algorithm_name = "pyopt"
        constraint_method = "ignore" # 'ignore','nan_on_unfeasible','constant_penalty_no_evaluation','algo_specific'
        verbose = True
        seed = 4
        np.random.seed(seed)

        def get_human_time() -> str:
            from datetime import datetime
            current_time = datetime.now()
            return current_time.strftime('%Y-%m-%d %H:%M:%S')

        filepath = f'results/data/{problem_name}_{algorithm_name}_{seed}.csv'
        def print_to_log(*args):
            with open(f"{filepath}.log", 'a') as f:
                print(*args,  file=f)


        print_to_log(f"Starting optimization {problem_name} {algorithm_name} {constraint_method} {seed} at {get_human_time()}")

        with open(filepath, "a") as f:
            print('evaluations;n_constraint_checks;time;f_best;x_best', file=f)

        prob = problem(problem_name, constraint_method)
        algo = optimization_algorithm(prob, algorithm_name, seed)

        f_best = 1e10
        x_best = None
        i = -1
        print_status_every = 30
        import time
        ref = time.time()
        while prob.n_f_evals < 10000:
            i += 1
            x = algo.ask()
            f = prob.f(x)
            if verbose:
                print_to_log("n_f_evals:", prob.n_f_evals, "n_constraint_checks:", prob.n_constraint_checks, "f:", f, "t:", time.time(), "x:", x.tolist())
            algo.tell(f)
            if f < f_best:
                f_best = f
                x_best = x
                print_to_log("--New best----------------------------------------------------")
                print_to_log(get_human_time(), f_best, x_best.tolist())
                print_to_log("--------------------------------------------------------------")
                with open(filepath, "a") as f:
                    print(f'{prob.n_f_evals};{prob.n_constraint_checks};{time.time() - ref};{f_best};{x_best.tolist()}', file=f)

            if i % print_status_every == 0 and not verbose:
                print_to_log("n_constraint_checks = ",prob.n_constraint_checks, "| n_f_evals", prob.n_f_evals, " | ", time.time() - ref ,"seconds")

        print_to_log("-------------------------------------------------------------")
        print_to_log("Finished local optimization.", get_human_time())
        print_to_log("n_f_evals:", prob.n_f_evals, "\nn_constraint_checks:", prob.n_constraint_checks, "\nx:", x.tolist(), "\nf:", f)
        print_to_log("Constraints: ")
        [print_to_log("g(x) = ", el) for el in  prob.constraint_check(x)]
        print_to_log("-------------------------------------------------------------")
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
