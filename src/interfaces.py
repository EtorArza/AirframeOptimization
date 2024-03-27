import numpy as np
import numpy.typing
import os
import time
from tqdm import tqdm as tqdm

problem_name_list = ["airframes", "windflo", "toy"]
algorithm_name_list = ["snobfit", "cobyqa", "pyopt", "nevergrad"]
constraint_method_list = ['ignore','nan_on_unfeasible','constant_penalty_no_evaluation','algo_specific', 'nn_encoding']

class problem:

    def __init__(self, problem_name:str, budget:int, constraint_method:str, seed:int):
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

        if problem_name == "airframes":
            import problem_airframes
            self.dim = 15
            self._constraint_check = problem_airframes.constraint_check_hexarotor_0_1
            self._f = lambda x: problem_airframes.f_symmetric_hexarotor_0_1(x)[0]
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

        self.n_constraints = len(self._constraint_check(np.random.random(self.dim)))
        if self.constraint_method == 'nn_encoding':
            import learn_encoding
            self.encoder = learn_encoding.solution_space_encoder(self, seed)


    def f(self, x:numpy.typing.NDArray[np.float_]):
        assert type(x) == np.ndarray

        return_value = None
        value_on_unfeasible = np.inf
        if self.constraint_method == 'ignore' or self.constraint_method =='algo_specific':
            return_value = 'f'
        elif self.constraint_method == 'nan_on_unfeasible':
            return_value = 'f' if np.all(np.array(self.constraint_check(x)) > 0) else np.nan
        elif self.constraint_method == 'constant_penalty_no_evaluation':
            return_value = 'f' if np.all(np.array(self.constraint_check(x)) > 0) else value_on_unfeasible
        elif self.constraint_method == 'nn_encoding':
            x_encoded = self.encoder.encode(x)
            if np.all(np.array(self.constraint_check(x)) > 0):
                self.n_f_evals += 1
                return self._f(x_encoded)
            else:
                self.n_unfeasible_on_ask += 1
                return np.inf
        else:
            raise ValueError("Constraint method "+str(self.constraint_method)+" not recognized.")

        if return_value=='f':
            return_value = self._f(x)
            self.n_f_evals+=1
        else:
            self.n_unfeasible_on_ask += 1
        return return_value

    def constraint_check(self, x:numpy.typing.NDArray[np.float_]) -> tuple:
        assert type(x) == np.ndarray
        self.n_constraint_evals+=1
        if self.constraint_method == 'nn_encoding':
            x_encoded = self.encoder.encode(x)
            res = self._constraint_check(x_encoded)
        else:
            res = self._constraint_check(x)
        assert type(res) == tuple, str(res) + " of type " + str(type(res))
        return res

    def plot_solution(self, x:numpy.typing.NDArray[np.float_]):
        pass

    def random_initial_sol(self, np_random_state) -> numpy.typing.NDArray[np.float_]:

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
            self.algo = algorithm_nevergrad.ng_optimizer(problem, seed, parallel_threads=1, budget=self.problem.budget)
        else:
            print("Algorithm name", algorithm_name, "not recognized.")

    def ask(self) -> numpy.typing.NDArray[np.float_]:
        self.algo.n_f_evals = self.problem.n_f_evals
        x = self.algo.ask()
        self.algo.n_f_evals = self.problem.n_f_evals
        assert type(x) == np.ndarray
        return x

    def tell(self, f) -> None:
        self.algo.tell(f)


def local_solve(problem_name, algorithm_name, constraint_method, seed, budget, reuse_encoding, log_every=None):

    optimization_cache_path_f = f'cache/{problem_name}_{algorithm_name}_{constraint_method}_{seed}_{budget}_f.npy'
    optimization_cache_path_x = f'cache/{problem_name}_{algorithm_name}_{constraint_method}_{seed}_{budget}_x.npy'
    prob_status_cache_path = f'cache/{problem_name}_{algorithm_name}_{constraint_method}_{seed}_{budget}_probstatus.txt'
    result_file_path = f'results/data/{problem_name}_{algorithm_name}_{constraint_method}_{seed}.csv'
    
    
    def save_prob_status(solver_time, f_best, n_f_evals, n_constraint_evals):
        with open(prob_status_cache_path, 'w') as file:
            file.write(str({'solver_time':solver_time, 'f_best':f_best, 'n_f_evals':n_f_evals, 'n_constraint_evals':n_constraint_evals,}))
    def load_prob_status():
        with open(prob_status_cache_path, 'r') as file:
            data = file.read()
            status = eval(data)
            return status
    def print_to_log(*args):
            with open(f"{result_file_path}.log", 'a') as f:
                print(*args,  file=f)

    def get_human_time() -> str:
        from datetime import datetime
        current_time = datetime.now()
        return current_time.strftime('%Y-%m-%d %H:%M:%S')

    np.random.seed(seed)

    prob = problem(problem_name, budget, constraint_method, 2 if reuse_encoding else seed) # Use same encoding with all seeds.
    algo = optimization_algorithm(prob, algorithm_name, seed)

    pb = tqdm(total=budget)


    # Load from cache (resume previous run)
    if os.path.exists(optimization_cache_path_f) or os.path.exists(optimization_cache_path_x) or os.path.exists(prob_status_cache_path):
        assert os.path.exists(optimization_cache_path_f) and os.path.exists(optimization_cache_path_x) and os.path.exists(prob_status_cache_path)
        print("Optimization cache files exist. Loading cached optimization...")
        x_list = [el for el in np.load(optimization_cache_path_x)]
        f_list = [el for el in np.load(optimization_cache_path_f)]
        assert len(x_list) == len(f_list)
        status = load_prob_status()
        prob.n_f_evals = status['n_f_evals']
        prob.n_constraint_evals = status['n_constraint_evals']
        solver_time = status['solver_time']
        if prob.n_f_evals >= budget:
            print("Error. Optimization already finished, n_f_evals > budget on load from cache.")
            exit(1)

        f_best = 1e10
        x_best = None
        for i in range(len(x_list)): # Repeat optimization algorithm with values from from cache.
            x = algo.ask()
            assert np.all(x == x_list[i]), f"cached solution {x} and algorithm solution {x_list[i]} at index {i} differ and should be exactly the same, the euclidean distance between them is {np.linalg.norm(x - x_list[i])}"
            f = f_list[i]
            algo.tell(f)
            if f < f_best:
                f_best = f
                x_best = x
        print("loaded.")
        print_to_log(f"--- Resuming optimization {problem_name} {algorithm_name} {constraint_method} {seed} {budget} at {get_human_time()}, from {prob.n_f_evals} evaluations.")


    # Start from scratch
    else:
        solver_time = 0.0
        f_best = 1e10
        x_best = None
        x_list = []
        f_list = []
        print_to_log(f"--- Starting optimization {problem_name} {algorithm_name} {constraint_method} {seed} {budget} at {get_human_time()}")
        with open(result_file_path, "a") as f:
            print('n_f_evals;n_constraint_evals;n_unfeasible_on_ask;time;f_best;x_best', file=f)


    assert len(x_list) == len(f_list), f"(len(x_list), len(f_list))={(len(x_list), len(f_list))}"


    i = 0
    ref = time.time() - solver_time
    while prob.n_f_evals < prob.budget:
        pb.n = prob.n_f_evals
        pb.refresh()
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

        if f < f_best:
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


    with open(result_file_path, "a") as file:
        print(f'{prob.n_f_evals};{prob.n_constraint_evals};{prob.n_unfeasible_on_ask};{time.time() - ref};{f_best};{x_best.tolist()}', file=file)

    print_to_log("-------------------------------------------------------------")
    print_to_log("Finished local optimization.", get_human_time())
    print_to_log("n_f_evals:", prob.n_f_evals, "\nn_constraint_evals:", prob.n_constraint_evals, "\nx:", x.tolist(), "\nf:", f)
    print_to_log("Constraints: ")
    [print_to_log("g(x) = ", el) for el in  prob.constraint_check(x)]
    print_to_log("-------------------------------------------------------------")
