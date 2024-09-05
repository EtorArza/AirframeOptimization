import isaacgym
import numpy as np
import numpy.typing
import os
import time
from tqdm import tqdm as tqdm
import numpy as np
import random
import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
import problem_airframes
import torch
import copy

def evaluate_airframe(x, train_seed, test_seed, task_info, n_repeated_evaluations):
    
    best_n_waypoints_per_reset = -1e8
    best_n_waypoints_reachable_based_on_battery_use = -1e8
    for i in range(n_repeated_evaluations):
        n_waypoints_per_reset, n_waypoints_reachable_based_on_battery_use = problem_airframes.f_symmetric_hexarotor_0_1(x, train_seed+20000*i, test_seed, task_info)
        best_n_waypoints_per_reset = max(best_n_waypoints_per_reset, n_waypoints_per_reset)
        best_n_waypoints_reachable_based_on_battery_use = max(best_n_waypoints_reachable_based_on_battery_use, n_waypoints_reachable_based_on_battery_use)

    f_res = {"n_waypoints_per_reset":best_n_waypoints_per_reset, "n_waypoints_reachable_based_on_battery_use":best_n_waypoints_reachable_based_on_battery_use}
    return f_res






class optimization_algorithm:

    def __init__(self, seed, ax_status_filepath, task_info):
        self.rs = np.random.RandomState(seed)
        self.ax_status_filepath = ax_status_filepath

        if not os.path.exists(ax_status_filepath):
            gs = GenerationStrategy(
                steps=[
                    GenerationStep(
                        model=Models.SOBOL,
                        num_trials=40,
                        min_trials_observed=3,
                        max_parallelism=1,
                        model_kwargs={"seed": self.rs.randint(int(1e6))},
                        model_gen_kwargs={},
                    ),
                    GenerationStep(
                        model=Models.BOTORCH_MODULAR,
                        num_trials=-1,
                        max_parallelism=1,
                        model_kwargs={"seed": self.rs.randint(int(1e6))},
                    ),
                ]
            )
            self.ax_client = AxClient()
            self.ax_client.create_experiment(
                parameters=[
                    {
                        "name": f"A_x{i:02d}",
                        "type": "range",
                        "value_type": "float",
                        "bounds": [0.0, 1.0],
                    } for i in range(15)
                ]+
                [
                    {
                        "name": f"B_motor{i:02d}",
                        "type": "range",
                        "value_type": "float",
                        "bounds": [0.0, 1.0]
                    } for i in range(3)
                ],
                objectives={
                    "n_waypoints_per_reset": ObjectiveProperties(minimize=False, threshold=task_info["threshold_n_waypoints_per_reset"]),
                    "n_waypoints_reachable_based_on_battery_use": ObjectiveProperties(minimize=False, threshold=task_info["threshold_n_waypoints_reachable_based_on_battery_use"]),
                }
            )
        else:
            self.ax_client = AxClient.load_from_json_file(ax_status_filepath)
            # best_x_prediction = self.ax_client.get_best_parameters()
            # best_x_prediction = [el[1] for el in sorted(best_x_prediction.items(), key=lambda z: float(z[0].strip("x")))]
            # print("Best x predicted by ax is: ", best_x_prediction)
            # exit(0)

        read_evaluations_from_log = False
        if read_evaluations_from_log:
            self.read_evaluations_from_log("results/data/offsetcone_6.csv.log")

    def read_evaluations_from_log(self, file_path):
        def read_log_file(file_path):
            known_evaluations = []
            with open(file_path, 'r') as file:
                for line in file:
                    if line.startswith("---"):
                        continue
                    if line.startswith("n_f_evals:"):
                        parts = line.split()
                        n_waypoints_per_reset = float(parts[3])
                        n_waypoints_reachable_based_on_battery_use = float(parts[5])

                        x_start = line.index('[')
                        x_string = line[x_start:]
                        x_values = eval(x_string)
                        
                        evaluation = {
                            "parameters": {
                                **{f"A_x{i:02d}": value for i, value in enumerate(x_values[:15])},
                                **{f"B_motor{i:02d}": value for i, value in enumerate(x_values[15:18])},
                            },
                            "objectives": {
                                "n_waypoints_per_reset": n_waypoints_per_reset,
                                "n_waypoints_reachable_based_on_battery_use": n_waypoints_reachable_based_on_battery_use
                            }
                        }
                        
                        known_evaluations.append(evaluation)

            return known_evaluations

        known_evaluations = read_log_file(file_path)
        for evaluation in tqdm(known_evaluations):
            _, idx = self.ax_client.attach_trial(evaluation["parameters"])
            print("Objectives: ", evaluation["objectives"])
            self.ax_client.complete_trial(trial_index=idx, raw_data=evaluation["objectives"])


    def ask(self):
        x_ax, self.trial_index = self.ax_client.get_next_trial()
        return np.array([el[1] for el in sorted(x_ax.items())])
        
    def tell(self, f_res):
        self.ax_client.complete_trial(trial_index=self.trial_index, raw_data=copy.deepcopy(f_res))

    def save_optimization_status(self, ax_status_filepath):
        self.ax_client.save_to_json_file(ax_status_filepath)




def local_solve(seed, budget, task_info):

    result_file_path = f'results/data/{task_info["waypoint_name"]}_{seed}.csv'
    ax_status_filepath = f'cache/ax_optimization_status/{task_info["waypoint_name"]}_{seed}.json'
    

    def print_to_log(*args):
            with open(f"{result_file_path}.log", 'a') as f:
                print(*args,  file=f)

    np.random.seed(seed+28342348)
    random.seed(seed+28342348)


    def get_human_time() -> str:
        from datetime import datetime
        current_time = datetime.now()
        return current_time.strftime('%Y-%m-%d %H:%M:%S')

    algo = optimization_algorithm(2, ax_status_filepath, task_info)

    solver_time = 0.0

    f_best = -1e10 if algo.ax_client.get_trials_data_frame().shape[0]==0 else algo.ax_client.get_trials_data_frame()["n_waypoints_per_reset"].max()
    x_best = None
    x_list = []
    f_list = []
    ref = time.time()

    for i in range(algo.ax_client.get_trials_data_frame().shape[0], budget):
        x = algo.ask()
        f = evaluate_airframe(x, i, 3, task_info, 4)
        algo.tell(f)
        algo.save_optimization_status(ax_status_filepath)
        df = algo.ax_client.get_trials_data_frame()

        if f["n_waypoints_per_reset"] > f_best:
            f_best = f["n_waypoints_per_reset"]
            x_best = x
            print_to_log("---New best----------------------------------------------------")
            print_to_log("---", get_human_time(), f_best, x_best.tolist())
            print_to_log("--------------------------------------------------------------")

        print_to_log("n_f_evals:", i, "n_waypoints_per_reset:", f["n_waypoints_per_reset"], "n_waypoints_reachable_based_on_battery_use:", f["n_waypoints_reachable_based_on_battery_use"], "t:", time.time() - ref, "x:", x.tolist())


    print_to_log("-------------------------------------------------------------")
    print_to_log("Finished local optimization.", get_human_time())
    print_to_log("-------------------------------------------------------------")


def airframe_repeatedly_train_and_enjoy(train_seed_list, enjoy_seed_list, max_epochs, pars, task_info, result_file_path):
    from airframes_objective_functions import motor_position_train, motor_position_enjoy, save_robot_pars_to_file, log_detailed_evaluation_results, model_to_onnx
    from problem_airframes import dump_animation_data_and_policy
    import os
    import torch
    import subprocess

    waypoint_name = task_info["waypoint_name"]
    save_robot_pars_to_file(pars)

    
    for seed_train in train_seed_list:
        t_start = time.time()
        evaluation_time = None
        exit_flag = motor_position_train(seed_train, max_epochs, waypoint_name, "position_setpoint_task" ,"headless")
        print("exit_flag", exit_flag)
        if exit_flag == "success":
            model_to_onnx()
            for seed_enjoy in enjoy_seed_list:
                for policy_path in [ "best_speed.onnx", "best_efficiency.onnx"]:
                    info_dict = motor_position_enjoy(seed_enjoy, policy_path, waypoint_name, "position_setpoint_task", "headless")
                    if evaluation_time is None:
                        evaluation_time = time.time() - t_start
                    log_detailed_evaluation_results(pars, policy_path, info_dict, seed_train, seed_enjoy, max_epochs, evaluation_time, result_file_path)
                    dump_animation_data_and_policy(pars, seed_train, seed_enjoy, info_dict, policy_path)

class problem_analyzer:

    def __init__(self, ax_json_status_path):
        self.ax_client = AxClient.load_from_json_file(ax_json_status_path)
        self.pareto_optimal_solutions = self.get_pareto_optimal_solutions_from_ax_client()
        self.pareto_optimal_solutions.sort(key=lambda el: (el[1]["n_waypoints_per_reset"],el[1]["n_waypoints_reachable_based_on_battery_use"]))

    def get_pareto_optimal_solutions_from_ax_client(self):
        processed_pars = []
        for solution_observed_f, solution_predicted_f in zip(self.ax_client.get_pareto_optimal_parameters(use_model_predictions=True).values(), self.ax_client.get_pareto_optimal_parameters(use_model_predictions=False).values()):
            x_observed = np.array([el[1] for el in sorted(solution_observed_f[0].items())])
            x_predicted = np.array([el[1] for el in sorted(solution_observed_f[0].items())])
            assert sum(abs(x_observed-x_predicted)) < 1e-6
            processed_pars.append([
                x_observed,
                solution_predicted_f[1][0],
                # solution_observed_f[1][0],
            ])
        return processed_pars

    def _crossover_x(self, x_a, x_b, closeness_to_b):
        assert 0.0 <= closeness_to_b <= 1.0
        assert len(x_a) == len(x_b) == 19

        from itertools import permutations
        def apply_permutation(x: np.ndarray, permu):
            assert 0 in permu and 1 in permu and 2 in permu and len(permu)==3
            x_res = x.copy()
            x_res[0:5] = x[(permu[0]*5):(permu[0]*5 +5)]
            x_res[15+0] = x[15+permu[0]]
            x_res[5:10] = x[(permu[1]*5):(permu[1]*5 +5)]
            x_res[15+1] = x[15+permu[1]]
            x_res[10:15] = x[(permu[2]*5):(permu[2]*5 +5)]
            x_res[15+2] = x[15+permu[2]]
            return x_res

        min_distance = 1e6
        for permu in list(permutations([0,1,2], 3)):
            x_b_permuted = apply_permutation(x_b, permu)
            if np.linalg.norm(x_a - x_b_permuted) < min_distance:
                min_distance = np.linalg.norm(x_a - x_b_permuted)
                best_x_b_permuted = x_b_permuted.copy()
        return x_a*(1.0 - closeness_to_b) + best_x_b_permuted*closeness_to_b

    def _get_relative_position_on_pareto(self, sol):
        f_sol = np.array([sol[1]["n_waypoints_per_reset"], sol[1]["n_waypoints_reachable_based_on_battery_use"]])
        f_pareto = []
        for el in self.pareto_optimal_solutions:
            f_pareto.append([el[1]["n_waypoints_per_reset"], el[1]["n_waypoints_reachable_based_on_battery_use"]])
        f_pareto = np.array(f_pareto)
        normalized_pareto = (f_pareto - np.min(f_pareto, axis=0)) / (np.max(f_pareto, axis=0) - np.min(f_pareto, axis=0))
        normalized_f = (f_sol - np.min(f_pareto, axis=0)) / (np.max(f_pareto, axis=0) - np.min(f_pareto, axis=0))
        res = (float(normalized_f[0] - normalized_f[1]) + 1.0) / 2.0
        assert 0 <= res <= 1
        return res

    def get_pareto_solutions_with_extra_interpolated_solutions(self):
        total_number_of_interpolated_solutions = 20
        res_interpolated_x = [self.pareto_optimal_solutions[0][0]]
        res_was_evaluated = [True]
        res_relative_pos_on_pareto = [0]
        last_solution = self.pareto_optimal_solutions[0]
        for solution in self.pareto_optimal_solutions[1:]:
            rel_pos_prev = self._get_relative_position_on_pareto(last_solution)
            rel_pos_current = self._get_relative_position_on_pareto(solution)
            assert rel_pos_current > rel_pos_prev
            for closeness_to_b in list(np.linspace(0,1,1+int(total_number_of_interpolated_solutions*(rel_pos_current - rel_pos_prev)), endpoint=False))[1:]:
                interpolated_x = self._crossover_x(last_solution[0], solution[0], closeness_to_b)
                res_interpolated_x.append(interpolated_x)
                res_was_evaluated.append(False)
                res_relative_pos_on_pareto.append(rel_pos_prev + closeness_to_b*(rel_pos_current - rel_pos_prev))
            res_interpolated_x.append(solution[0])
            res_was_evaluated.append(True)
            res_relative_pos_on_pareto.append(rel_pos_current)
            last_solution = solution
        return res_interpolated_x, res_was_evaluated, res_relative_pos_on_pareto



