import numpy as np
import numpy.typing
import os
import time
from tqdm import tqdm as tqdm

problem_name_list = ["airframes", "windflo", "toy"]
algorithm_name_list = ["snobfit", "cobyqa", "pyopt", "nevergrad", "scipy", "skoptbo"]
constraint_method_list = ['ignore','nan_on_unfeasible','constant_penalty_no_evaluation','algo_specific', 'nn_encoding']

class problem:

    def __init__(self, problem_name:str, budget:int, constraint_method:str, seed:int, reuse_encoding:bool):
        '''
        Initializes the problem to be solved. All the problems are minimization of the objective 
        function f, and solutions are feasible as long as the constraints >= 0. This is the convention used 
        in the textbook Numerical optimization by Jorge Nocedal.

        
        Parameters:
            problem_name (str): The name of the problem.
            constraint_method (str): How to deal with constraints. Supported values are 'ignore','nan_on_unfeasible', 'constant_penalty_no_evaluation', 'algo_specific'.
        '''
        assert problem_name in problem_name_list
        assert constraint_method in constraint_method_list
        self.constraint_method = constraint_method
        self.problem_name = problem_name
        self.budget = budget
        self.n_f_evals = 0
        self.n_constraint_evals = 0
        self.n_unfeasible_on_ask = 0
        self.n_f_calls_without_evaluation = 0
        self.x0 = None
        self.x0_f = None
        self.rs = np.random.RandomState(seed=seed+128428)

        if problem_name == "airframes":
            import problem_airframes
            self.dim = 15
            self._constraint_check = lambda x: (1.0, 1.0) #problem_airframes.constraint_check_hexarotor_0_1
            self._f = lambda x: problem_airframes.f_symmetric_hexarotor_0_1(x, self.rs.randint(1e8))[2]
            self.plot_solution = lambda x: problem_airframes.plot_airframe_design(problem_airframes._decode_symmetric_hexarotor_to_RobotParameter(x))

        if problem_name == "windflo":
            import problem_windflo
            self.dim = problem_windflo.SOLUTION_DIM
            self._constraint_check = problem_windflo.constraint_check
            self._f = problem_windflo.f
            self.plot_solution = problem_windflo.plot_WindFLO

        if problem_name == "toy":
            dim = 8
            import problem_toy
            self.dim = dim
            self._constraint_check = problem_toy.constraint_check
            self._f = problem_toy.f

        self.last_x_constraint_check = np.random.random(self.dim)
        self.last_x_constraint_check_result = self._constraint_check(self.last_x_constraint_check)
        self.n_constraints = len(self.last_x_constraint_check_result)

        if self.constraint_method == 'nn_encoding':
            import learn_encoding
            self.encoder = learn_encoding.solution_space_encoder(self, 2 if reuse_encoding else seed)


    def f(self, x:numpy.typing.NDArray[np.float_]):
        assert type(x) == np.ndarray

        if self.constraint_method in  ('ignore','algo_specific'):
            return_value = 'f'
        elif self.constraint_method == 'nan_on_unfeasible':
            return_value = 'f' if np.all(np.array(self.constraint_check(x)) > 0) else np.nan
        elif self.constraint_method == 'constant_penalty_no_evaluation':
            return_value = 'f' if np.all(np.array(self.constraint_check(x)) > 0) else np.inf
        elif self.constraint_method == 'nn_encoding':
            return_value = 'f' if np.all(np.array(self.constraint_check(x)) > 0) else np.inf # encode() applied directly in self.constraint_check()
        else:
            raise ValueError("Constraint method "+str(self.constraint_method)+" not recognized.")

        # This check is just to measure n_unfeasible_on_ask for plotting purposes, and has nothing to do with 
        # optimization (that is why we do not count the n_constraint_evals).
        # Need to do the extra check because 'ignore' and 'algo specific' might evaluate unfeasible solutions.
        self.n_constraint_evals -= 1  
        if not np.all(np.array(self.constraint_check(x)) > 0):
            self.n_unfeasible_on_ask += 1

        if return_value=='f':
            if self.constraint_method == 'nn_encoding':
                x_encoded = self.encoder.encode(x)
            else:
                x_encoded = x

            return_value = self._f(x_encoded)
            self.n_f_evals+=1
            self.n_f_calls_without_evaluation = 0
        else:
            self.n_f_calls_without_evaluation += 1
            if self.n_f_calls_without_evaluation > 2000:
                self.n_f_evals += 1
        return return_value

    def constraint_check(self, x:numpy.typing.NDArray[np.float_]) -> tuple:
        assert type(x) == np.ndarray
        if self.constraint_method == 'nn_encoding':
            x_encoded = self.encoder.encode(x)
        else:
            x_encoded = x

        if np.all(self.last_x_constraint_check == x):
            res = self.last_x_constraint_check_result
        else:
            self.last_x_constraint_check = x
            res = self._constraint_check(x_encoded)
            self.n_constraint_evals += 1
            self.last_x_constraint_check_result = res
        assert type(res) == tuple, str(res) + " of type " + str(type(res))
        return res

    def plot_solution(self, x:numpy.typing.NDArray[np.float_]):
        pass

    def random_initial_sol(self) -> numpy.typing.NDArray[np.float_]:

        if self.constraint_method in ('ignore'):
            return self.rs.random(self.dim)
        elif self.constraint_method in ('nan_on_unfeasible', 'constant_penalty_no_evaluation', 'algo_specific', 'nn_encoding'):
            feasible = False
            while not feasible:
                self.x0 = self.rs.random(self.dim)
                feasible = np.all(np.array(self.constraint_check(self.x0)) > 0)
            return self.x0
        else:
            raise ValueError("Constraint method "+str(self.constraint_method)+" not recognized.")




class optimization_algorithm:

    def __init__(self, problem: problem, algorithm_name, seed):
        assert algorithm_name in algorithm_name_list
        self.problem = problem
        self.algorithm_name = algorithm_name
        if algorithm_name == "snobfit":
            import algorithm_snobfit
            self.algo = algorithm_snobfit.snobfit(problem, seed)
        elif algorithm_name == "cobyqa":
            import algorithm_cobyqa
            self.algo = algorithm_cobyqa.cobyqa(problem, seed)
        elif algorithm_name == "pyopt":
            import algorithm_pyopt
            self.algo = algorithm_pyopt.pyopt(problem, seed)
        elif algorithm_name == "nevergrad":
            import algorithm_nevergrad
            self.algo = algorithm_nevergrad.ng_optimizer(problem, seed, parallel_threads=1, total_budget=self.problem.budget)
        elif algorithm_name == "scipy":
            import algorithm_scipy
            self.algo = algorithm_scipy.scipy_optimizer(problem, seed)
        elif algorithm_name == "skoptbo":
            import algorithm_skoptbo
            self.algo = algorithm_skoptbo.skoptbo_optimizer(problem, seed)
        else:
            print("Algorithm name", algorithm_name, "not recognized.")

    def ask(self) -> numpy.typing.NDArray[np.float_]:
        if not self.problem.x0 is None:
            return self.problem.x0

        self.algo.n_f_evals = self.problem.n_f_evals
        x = self.algo.ask()
        self.algo.n_f_evals = self.problem.n_f_evals
        assert type(x) == np.ndarray
        assert max(x) <= 1.0 and min(x) >= 0.0, f"x = {x} out of bounds [0,1]"
        x += self.problem.rs.normal(0, 1e-9, size=self.problem.dim)
        x = np.clip(x, a_min=0.0, a_max=1.0)
        return x

    def tell(self, f) -> None:
        if not self.problem.x0 is None:
            self.problem.x0 = None
            return
        self.algo.tell(f)



def _resume_previous_local_solve(prob: problem, algo:optimization_algorithm, optimization_cache_path_f, optimization_cache_path_x, prob_status_cache_path):
    x_list = [el for el in np.load(optimization_cache_path_x)]
    f_list = [el for el in np.load(optimization_cache_path_f)]
    assert len(x_list) == len(f_list)

    def load_prob_status():
        with open(prob_status_cache_path, 'r') as file:
            data = file.read()
            status = eval(data)
            return status

    status = load_prob_status()
    prob.n_f_evals = status['n_f_evals']
    prob.n_constraint_evals = status['n_constraint_evals']
    solver_time = status['solver_time']
    if prob.n_f_evals >= prob.budget:
        print("Error. Optimization already finished, n_f_evals > budget on load from cache.")
        print("Skipping optimization.")
        return [None,]*5

    f_best = 1e10
    x_best = None
    for i in range(len(x_list)): # Repeat optimization algorithm with values from from cache.
        x = algo.ask()


        print("-")
        print(i)
        print(x_list[i])
        print(x)
        print("-")

        assert np.all(x == x_list[i]), f"cached solution {x_list[i]} and algorithm solution {x} at index {i} differ and should be exactly the same, the euclidean distance between them is {np.linalg.norm(x - x_list[i])}"
        f = f_list[i]
        algo.tell(f)
        if f < f_best and (prob.constraint_method == 'ignore' or np.all(np.array(prob.constraint_check(x)) > 0)):
            f_best = f
            x_best = x
    ref = time.time() - solver_time
    print("loaded.")
    return ref, x_best, f_best, x_list, f_list



def local_solve(problem_name, algorithm_name, constraint_method, seed, budget, reuse_encoding, log_every=None):

    optimization_cache_path_f = f'cache/{problem_name}_{algorithm_name}_{constraint_method}_{seed}_{budget}_f.npy'
    optimization_cache_path_x = f'cache/{problem_name}_{algorithm_name}_{constraint_method}_{seed}_{budget}_x.npy'
    prob_status_cache_path = f'cache/{problem_name}_{algorithm_name}_{constraint_method}_{seed}_{budget}_probstatus.txt'
    result_file_path = f'results/data/{problem_name}_{algorithm_name}_{constraint_method}_{seed}.csv'
    np.random.seed(seed+28342348)
    import random
    random.seed(seed+28342348)

    def save_prob_status(solver_time, f_best, n_f_evals, n_constraint_evals):
        with open(prob_status_cache_path, 'w') as file:
            file.write(str({'solver_time':solver_time, 'f_best':f_best, 'n_f_evals':n_f_evals, 'n_constraint_evals':n_constraint_evals,}))
    def print_to_log(*args):
            with open(f"{result_file_path}.log", 'a') as f:
                print(*args,  file=f)

    def get_human_time() -> str:
        from datetime import datetime
        current_time = datetime.now()
        return current_time.strftime('%Y-%m-%d %H:%M:%S')


    prob = problem(problem_name, budget, constraint_method, seed, reuse_encoding)
    algo = optimization_algorithm(prob, algorithm_name, seed)

    # pb = tqdm(total=budget)


    # Load from cache (resume previous run)
    if os.path.exists(optimization_cache_path_f) or os.path.exists(optimization_cache_path_x) or os.path.exists(prob_status_cache_path):
        assert os.path.exists(optimization_cache_path_f) and os.path.exists(optimization_cache_path_x) and os.path.exists(prob_status_cache_path)
        print("Optimization cache files exist. Loading cached optimization...")
        ref, x_best, f_best, x_list, f_list = _resume_previous_local_solve(prob, algo, optimization_cache_path_f, optimization_cache_path_x, prob_status_cache_path)
        if ref is None: # Skip finished optimization
            return
        print_to_log(f"--- Resuming optimization {problem_name} {algorithm_name} {constraint_method} {seed} {budget} at {get_human_time()}, from {prob.n_f_evals} evaluations.")


    # Start from scratch
    else:
        solver_time = 0.0
        f_best = 1e10
        x_best = None
        x_list = []
        f_list = []
        ref = time.time()
        print_to_log(f"--- Starting optimization {problem_name} {algorithm_name} {constraint_method} {seed} {budget} at {get_human_time()}")
        if os.path.exists(result_file_path):
            raise FileExistsError(f"result file {result_file_path} already exists, but no cache file was loaded.")
        else:
            with open(result_file_path, "a") as f:
                print('n_f_evals;n_constraint_evals;n_unfeasible_on_ask;time;f_best;x_best', file=f)



    assert len(x_list) == len(f_list), f"(len(x_list), len(f_list))={(len(x_list), len(f_list))}"


    i = 0
    while prob.n_f_evals < prob.budget:
        # pb.n = prob.n_f_evals
        # pb.refresh()
        x = algo.ask()
        x_list.append(x)
    
        f = prob.f(x)
        f_list.append(f)
        solver_time = time.time() - ref
        algo.tell(f)

        # save optimization status to cache
        np.save(optimization_cache_path_x, np.array(x_list))
        np.save(optimization_cache_path_f, np.array(f_list))
        save_prob_status(solver_time, f_best, prob.n_f_evals, prob.n_constraint_evals)

        if f < f_best and (prob.constraint_method == 'ignore' or np.all(np.array(prob.constraint_check(x)) > 0)):
            f_best = f
            x_best = x
            print_to_log("---New best----------------------------------------------------")
            print_to_log("---", get_human_time(), f_best, x_best.tolist())
            print_to_log("--------------------------------------------------------------")
            with open(result_file_path, "a") as file:
                print(f'{prob.n_f_evals};{prob.n_constraint_evals};{prob.n_unfeasible_on_ask};{time.time() - ref};{f_best};{x_best.tolist()}', file=file)

        if not log_every is None and i % log_every == 0:
            print_to_log("n_f_evals:", prob.n_f_evals, "n_constraint_evals:", prob.n_constraint_evals, "f:", f, "t:", time.time() - ref, "x:", x.tolist())
        i += 1

    # pb.close()
    with open(result_file_path, "a") as file:
        print(f'{prob.n_f_evals};{prob.n_constraint_evals};{prob.n_unfeasible_on_ask};{time.time() - ref};{f_best};{x_best.tolist()}', file=file)

    print_to_log("-------------------------------------------------------------")
    print_to_log("Finished local optimization.", get_human_time())
    print_to_log("n_f_evals:", prob.n_f_evals, "\nn_constraint_evals:", prob.n_constraint_evals, "\nx:", x_best.tolist(), "\nf:", f_best)
    print_to_log("Constraints: ")
    [print_to_log("g(x) = ", el) for el in  prob.constraint_check(x_best)]
    print_to_log("-------------------------------------------------------------")
