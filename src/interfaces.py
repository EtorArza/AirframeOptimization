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

def evaluate_airframe(x, train_seed, test_seed, task_info):
    info_dict = problem_airframes.f_symmetric_hexarotor_0_1(x, train_seed, test_seed, task_info)
    if info_dict is None:
        f_res = {"nWaypointsReached/nResets":0.0, "total_energy/nWaypointsReached":1e6}
    else:
        f_res = {"nWaypointsReached/nResets":(info_dict['f_nWaypointsReached']/info_dict['f_nResets']).cpu().item(), "total_energy/nWaypointsReached":(info_dict['f_total_energy']/torch.clamp(info_dict['f_nWaypointsReached'], min=1.0)).cpu().item()}
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
                        num_trials=100,
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
                ]+
                [
                    {
                        "name": f"C_battery",
                        "type": "range",
                        "value_type": "float",
                        "bounds": [0.0, 1.0]
                    }
                ],
                objectives={
                    "nWaypointsReached/nResets": ObjectiveProperties(minimize=False, threshold=task_info["threshold_nWaypointsReached/nResets"]),
                    "total_energy/nWaypointsReached": ObjectiveProperties(minimize=True, threshold=task_info["threshold_total_energy/nWaypointsReached"]),
                }
            )
        else:
            self.ax_client = AxClient.load_from_json_file(ax_status_filepath)
            # best_x_prediction = self.ax_client.get_best_parameters()
            # best_x_prediction = [el[1] for el in sorted(best_x_prediction.items(), key=lambda z: float(z[0].strip("x")))]
            # print("Best x predicted by ax is: ", best_x_prediction)
            # exit(0)


    def ask(self):
        x_ax, self.trial_index = self.ax_client.get_next_trial()
        return np.array([el[1] for el in sorted(x_ax.items())])
        
    def tell(self, f_res):
        self.ax_client.complete_trial(trial_index=self.trial_index, raw_data=copy.deepcopy(f_res))

    def save_optimization_status(self, ax_status_filepath):
        self.ax_client.save_to_json_file(ax_status_filepath)




def local_solve(seed, budget, task_info):

    task_info_str = "" if task_info is None else task_info["waypoint_name"]
    result_file_path = f'results/data/{task_info_str}_{seed}.csv'
    ax_status_filepath = f'cache/ax_optimization_status/{task_info_str}_{seed}.csv'
    

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

    f_best = -1e10 if algo.ax_client.get_trials_data_frame().shape[0]==0 else algo.ax_client.get_trials_data_frame()["nWaypointsReached/nResets"].max()
    x_best = None
    x_list = []
    f_list = []
    ref = time.time()

    for i in range(algo.ax_client.get_trials_data_frame().shape[0], budget):
        x = algo.ask()
        f = evaluate_airframe(x, i, 3, task_info)
        algo.tell(f)
        algo.save_optimization_status(ax_status_filepath)
        df = algo.ax_client.get_trials_data_frame()

        if f["nWaypointsReached/nResets"] > f_best:
            f_best = f["nWaypointsReached/nResets"]
            x_best = x
            print_to_log("---New best----------------------------------------------------")
            print_to_log("---", get_human_time(), f_best, x_best.tolist())
            print_to_log("--------------------------------------------------------------")

        print_to_log("n_f_evals:", i, "nWaypointsReached/nResets:", f["nWaypointsReached/nResets"], "total_energy/nWaypointsReached:", f["total_energy/nWaypointsReached"], "t:", time.time() - ref, "x:", x.tolist())


    print_to_log("-------------------------------------------------------------")
    print_to_log("Finished local optimization.", get_human_time())
    print_to_log("-------------------------------------------------------------")


def airframe_repeatedly_train_and_enjoy(train_seed_list, enjoy_seed_list, max_epochs, pars, task_info, result_file_path):
    from airframes_objective_functions import motor_position_train, motor_position_enjoy, save_robot_pars_to_file, log_detailed_evaluation_results, model_to_onnx
    from problem_airframes import loss_function, dump_animation_data_and_policy
    import os
    import torch
    import subprocess

    waypoint_name = task_info["waypoint_name"]
    resfilename = f"results/data/hex_repeatedly_train_{max_epochs}s_{waypoint_name}.csv"
    save_robot_pars_to_file(pars)

    
    for seed_train in train_seed_list:
        exit_flag = motor_position_train(seed_train, max_epochs, True, waypoint_name)
        print("exit_flag", exit_flag)
        if exit_flag == "success":
            model_to_onnx()
            for seed_enjoy in enjoy_seed_list:
                info_dict = motor_position_enjoy(seed_enjoy, True, waypoint_name)
                f = loss_function(info_dict)
                log_detailed_evaluation_results(pars, info_dict, seed_train, seed_enjoy, max_epochs, result_file_path)
                dump_animation_data_and_policy(pars, seed_train, seed_enjoy, info_dict)