import sys
from tqdm import tqdm as tqdm
import numpy as np


task_info = {
    "waypoint_name": "lrcontinuous",
    "threshold_n_waypoints_per_reset": 4.0,
    "threshold_n_waypoints_reachable_based_on_battery_use": 200.0
}

selected_designs = {
    "s_000":[0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5, 0.00, 0.00, 0.00],
    "s_111":[0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5, 0.25, 0.25, 0.25],
    "s_222":[0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5, 0.50, 0.50, 0.50],
    "s_333":[0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5, 0.70, 0.70, 0.70],
    "s_444":[0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5, 0.99, 0.99, 0.99],
    "GP_posterior":[0.0, 0.42335352141207966, 0.4412499403717871, 1.0, 0.25947356154327933, 0.4122988001891047, 0.5814174731889221, 0.27038851248100043, 0.655373970654569, 0.10915401142834615, 0.16081482051639673, 0.6025436654695269, 0.8707279852493297, 0.6162662254436971, 0.7930472489877861, 0.2979701844186879, 0.46131302767998633, 0.4595460628161921],
    "observed":[0.0, 0.4546179012055345, 0.43336229440185176, 1.0, 0.09095459403859071, 0.40196376344042195, 0.5792803462063196, 0.2990637845402627, 0.656236183203457, 0.10150161933277721, 0.004472994781259263, 0.612007834024074, 0.8586475028922177, 0.6005588620661254, 0.7924056990140599, 0.2701273765396683, 0.5718815532877707, 0.3881165320219997],
}


if __name__ == "__main__":

    # Show the percentage of feasible solutions in a problem
    if sys.argv[1] == "--venn-diagram":
        assert sys.argv[2] in problem_name_list
        import plot_src
        problem_name = sys.argv[2]
        n_montecarlo = 5000
        prob = problem(problem_name, 100, 'ignore', 2, True)
        set_list = [set() for _ in range(prob.n_constraints)]
        for i in tqdm(range(n_montecarlo)):
            x_random = np.random.random(prob.dim)
            res = prob.constraint_check(x_random)
            [set_list[idx].add(i) for idx in range(prob.n_constraints) if res[idx] > 0.0]
        plot_src.plot_venn_diagram(problem_name, set_list, n_montecarlo, ["constraint_"+str(i) for i in range(prob.n_constraints)])

    # Directly solve problem locally, with f function that returns np.nan on infeasible solutions.
    elif sys.argv[1] == "--local-solve":
        from interfaces import local_solve
        from airframes_objective_functions import update_task_config_parameters
        sys.argv.pop()
        seed = 6
        budget = 600
        local_solve(seed, budget, task_info)

    # Directly solve problem locally, with f function that returns np.nan on infeasible solutions.
    elif sys.argv[1] == "--all-local-solve":
        from interfaces import *
        sys.argv.pop()
        reuse_encoding = True
        budget = 50
        global_log_path = "global_log.log"
        if os.path.exists(global_log_path):
            os.remove(global_log_path)

        algorithm_name_list = ["nevergrad",] #"snobfit", "cobyqa", "pyopt"]:
        constraint_method_list = ["ignore"]# ["algo_specific", "nn_encoding", 'constant_penalty_no_evaluation']
        problem_name_list = ['airframes']#['windflo', 'toy']
        seed_list = list(range(2,102))

        pb = tqdm(total = len(algorithm_name_list)*len(constraint_method_list)*len(problem_name_list)*len(seed_list))
        for algorithm_name in algorithm_name_list:
            for constraint_method in constraint_method_list:
                for problem_name in problem_name_list:
                    for seed in seed_list:
                        with open(global_log_path, "a") as file:
                            file.write(f"Launching {problem_name} {seed} {constraint_method} - ")
                        local_solve(problem_name, algorithm_name, constraint_method, seed, budget, reuse_encoding, log_every=100)
                        with open(global_log_path, "a") as file:
                            file.write("done\n")
                        pb.update()
        pb.close()

    elif sys.argv[1] == "--plot-rotor-properties":

        from aerial_gym_dev.utils.battery_rotor_dynamics import manufacturerComponentData, BatteryRotorDynamics
        from aerial_gym_dev.utils.custom_math import linear_1d_interpolation
 
        component_data = manufacturerComponentData("cpu")
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        import torch
        import os

        battery_S = 4
        compatible_motors, compatible_batteries = BatteryRotorDynamics.get_compatible_battery_and_motors_indices(battery_S)


        os.makedirs('results/figures/motor_properties', exist_ok=True)
        component_data = manufacturerComponentData("cpu")


        num_combos = len(compatible_motors)
        num_rpms = component_data.rps.shape[1]

        forces = np.zeros((num_combos, num_rpms))
        efficiencies = np.zeros((num_combos, num_rpms))
        labels = []
        for i in range(num_combos):
            force = component_data.motor_dict[compatible_motors[i]]["force"].numpy()
            currents = component_data.currents[compatible_motors[i], :].numpy()
            
            forces[i, :] = force
            efficiencies[i, :] = force / currents
            labels.append(component_data.motor_dict[compatible_motors[i]]["name"])


        forces = forces[:, ~np.isnan(forces).any(axis=0)]
        efficiencies = efficiencies[:, ~np.isnan(efficiencies).any(axis=0)]



        plt.figure(figsize=(10, 4))
        sns.heatmap(forces, cmap="viridis", annot=False, fmt=".2f", cbar_kws={'label': 'Force (N)'})
        plt.title("Motor-Propeller Combo Forces")
        plt.xlabel("RPM Index")
        plt.ylabel("Motor-Propeller Combo Index")
        plt.yticks(np.arange(num_combos) + 0.5, labels, rotation=0)
        plt.tight_layout()
        plt.savefig('results/figures/motor_properties/forces.pdf')
        plt.close()

        # Create the efficiency heatmap
        plt.figure(figsize=(10, 4))
        sns.heatmap(efficiencies, cmap="viridis", annot=False, fmt=".2f", cbar_kws={'label': 'Efficiency (N/A)'})
        plt.title("Motor-Propeller Combo Efficiencies")
        plt.yticks(np.arange(num_combos) + 0.5, labels, rotation=0)
        plt.xlabel("RPM Index")
        plt.ylabel("Motor-Propeller Combo Index")
        plt.tight_layout()
        plt.savefig('results/figures/motor_properties/efficiency.pdf')
        plt.close()

        num_force_points = 100
        interpolated_efficiencies = np.full((num_combos, num_force_points), np.nan)
        max_force = np.max(forces)
        x_new = np.linspace(0, max_force, num_force_points)

        for i in range(num_combos):
            x_tensor = torch.tensor(forces[i], dtype=torch.float32)
            y_tensor = torch.tensor(efficiencies[i], dtype=torch.float32)
            xnew_tensor = torch.tensor(x_new, dtype=torch.float32)
            
            # Interpolate only up to the maximum force for this combo
            max_force_index = np.searchsorted(x_new, np.max(forces[i]), side='right')
            interpolated_values = linear_1d_interpolation(x_tensor, y_tensor, xnew_tensor[:max_force_index]).numpy()
            
            interpolated_efficiencies[i, :max_force_index] = interpolated_values

        plt.figure(figsize=(16, 8))
        sns.heatmap(interpolated_efficiencies, cmap="viridis", annot=False, fmt=".2f",
                    cbar_kws={'label': 'Efficiency (N/A)'}, mask=np.isnan(interpolated_efficiencies))
        plt.title("Efficiency with respect to Force")
        plt.yticks(np.arange(num_combos) + 0.5, labels, rotation=0)
        plt.xlabel("Force (N)")
        plt.ylabel("Motor-Propeller Combo Index")
        n_xticklabels = 8
        x_tick_positions = [(i*(num_force_points-1)) // (n_xticklabels-1) for i in range(n_xticklabels)]
        print(x_tick_positions)
        plt.xticks(x_tick_positions, [f"{x_new[x_tick_positions[i]]:.1f}" for i in range(n_xticklabels)])
        plt.tight_layout()
        plt.savefig('results/figures/motor_properties/efficiency_wrt_force.pdf')
        plt.close()
        print("done!")

    elif sys.argv[1] == "--hex-different-epochs":
        from interfaces import *
        from problem_airframes import loss_function, dump_animation_data_and_policy, from_0_1_to_RobotParameter
        import itertools

        hex_pars = from_0_1_to_RobotParameter(np.array(
            [0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5] 
        ))


        train_seed_list = list(range(3,18))
        enjoy_seed_list = list(range(42,44))
        pars = hex_pars
        max_epochs_list = [1000, 2000, 4000]
        
        for max_epochs in max_epochs_list:
            airframe_repeatedly_train_and_enjoy(train_seed_list, enjoy_seed_list, max_epochs, pars, task_info, f"results/data/repeatedly_standard_hex_different_train_seed_{task_info['waypoint_name']}.csv")

    elif sys.argv[1] == "--airframes-repeatedly-train":
        from problem_airframes import from_0_1_to_RobotParameter
        from airframes_objective_functions import update_task_config_parameters
        from interfaces import *
        import itertools
        

        train_seed_list = list(range(1001,1020))
        enjoy_seed_list = list(range(42,44))
        max_epochs = 4000
        
        update_task_config_parameters(0,False, task_info["waypoint_name"], False)
        for train_seed_sublist in [[s] for s in train_seed_list]:
            for pars_name, x  in selected_designs.items():
                if pars_name in ("s_000","s_111","s_222","s_333","s_444") and train_seed_sublist[0] <= 1009:
                    continue

                x = np.array(x)
                x_0_1 = x[:15]
                motor_idx_0_1 = x[15:18]
            
                pars = from_0_1_to_RobotParameter(x_0_1, motor_idx_0_1)
                print(pars)
                pars.pars_name = pars_name
                airframe_repeatedly_train_and_enjoy(train_seed_sublist, enjoy_seed_list, max_epochs, pars, task_info, f"results/data/repeatedly_train_chosen_designs_{task_info['waypoint_name']}.csv")

    elif sys.argv[1] == "--plot-repeatedly-train":
        import plot_src
        plot_src.boxplots_repeatedly_different_train_seed(f"results/data/repeatedly_train_chosen_designs_{task_info['waypoint_name']}.csv")
        exit(0)
        plot_src.generate_bokeh_interactive_plot(f"results/data/local_solve_{task_info['waypoint_name']}.csv", f"{task_info['waypoint_name']}")
        # plot_src.boxplots_repeatedly_different_train_seed(f"results/data/repeatedly_standard_hex_different_train_seed_{task_info['waypoint_name']}.csv", task_info['waypoint_name'])
        # plot_src.multiobjective_scatter_by_train_time(f"results/data/details_every_evaluation_{waypoint_name}.csv")
        exit(0)

    elif sys.argv[1] == "--learn-hover-policy":
        from airframes_objective_functions import get_hover_policy
        file_path = "cache/airframes_animationdata/778443724_210_3_leftright_best_speed_airframeanimationdata.wb" 
        get_hover_policy(file_path, 6, "visualize")

    elif sys.argv[1] == "--plot-rank-error":
        from plot_src import *
        plot_accuracy_loss_vs_training_time("f_2_observed.txt")

    elif sys.argv[1] == "--evaluate-one":
        from problem_airframes import *
        # # Best solution
        # x = np.array([
        #     0.4505677118451259, 0.3790109151931583, 0.7553272392295209, 0.0, 0.017070865111953607,
        #     1.0, 0.4826375013108367, 0.4808966415561724, 0.04596926511532326, 0.09570130887176045, 
        #     0.3845583658094259, 0.5507515765703298, 0.6896674472409062, 0.44108011522818646, 0.0, 
        #     0.12127944102870869, 0.5957224152246153, 0.0
        # ])

        # hex       
        x = np.array([
                    0.0, 0.5, 0.5000, 0.5, 0.5, 
                    0.0, 0.5, 0.1667, 0.5, 0.5, 
                    0.0, 0.5, 0.8333, 0.5, 0.5,
                    0.0, 0.0, 0.0,
                    ])
        
        # # quad
        # x = np.array([0.0, 0.5, 0.25, 0.5, 0.5, 
        #               0.0, 0.5, 0.75, 0.5, 0.5, 
        #              ])

        x_0_1 = x[:15]
        motor_idx_0_1 = x[15:18]
        pars = from_0_1_to_RobotParameter(x_0_1, motor_idx_0_1)
        save_robot_pars_to_file(pars)
        plot_airframe_to_file_isaacgym(pars, filepath="test_airframe_render.png")
        # plot_admisible_set(pars)

        save_robot_pars_to_file(pars)
        # plot_airframe_to_file_isaacgym(pars, filepath="demo_image.png")

        seed_train = 610
        seed_enjoy = 3
        start = time.time()
        motor_rl_objective_function(pars, seed_train, seed_enjoy, 2000, task_info["waypoint_name"], "problem_airframes_train_and_enjoy.csv", render="headless")
        exit(0)

    elif sys.argv[1] == "--enjoy-one":
        from problem_airframes import *

        # # Optimized
        # file_path = "cache/airframes_animationdata/3709956583_1009_42_lrcontinuous_best_speed_airframeanimationdata.wb" 
        # s_111
        file_path = "cache/airframes_animationdata/4134737168_610_3_lrcontinuous_best_speed_airframeanimationdata.wb" 

        animation_data  = load_animation_data_and_policy(file_path) # load policy into correct path
        save_robot_pars_to_file(animation_data["pars"])
        plot_airframe_to_file_isaacgym(animation_data["pars"], filepath="test_airframe_render.png")
        motor_position_enjoy(3, animation_data["policy_path"], task_info["waypoint_name"], "position_setpoint_task", "headless")


    elif sys.argv[1] == "--ax-get-conclusions-solution-space":
        import plot_src
        from interfaces import problem_analyzer
        pa = problem_analyzer(f"cache/ax_optimization_status/{task_info['waypoint_name']}_6.json")

        exit(0)
        interpolated_x, was_evaluated, relative_pos_on_pareto = pa.get_pareto_solutions_with_extra_interpolated_solutions()
        plot_src.animate_solution_interpolation(interpolated_x, was_evaluated, relative_pos_on_pareto)

    else:
        print("sys.argv[1]=",sys.argv[1],"not recognized.", sep=" ")
