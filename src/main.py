import sys
from tqdm import tqdm as tqdm
import numpy as np


task_info = {
    "waypoint_name": "leftright",
    "threshold_n_waypoints_per_reset": 4.0,
    "threshold_n_waypoints_reachable_based_on_battery_use": 200.0
}

selected_designs = {
    # "s_111":[0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5, 0.25, 0.25, 0.25],
    # "s_222":[0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5, 0.5, 0.5, 0.5],
    # "s_333":[0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5, 0.7, 0.7, 0.7],
    "best_GP_posterior":[0.0, 0.22866268504650217, 0.25536810175970565, 1.0, 0.9956485526782684, 0.21323650495472396, 0.53182992745159, 0.8631856188621372, 0.6316025675611324, 1.0, 0.7500653690969006, 0.5714853674056419, 0.3421233215412284, 0.0, 0.35179863097711594, 0.1099251368417103, 0.6465696431830745, 0.7254261187678717],
    "best_observed":[0.028446687650210897, 0.2524035927519064, 0.254165546000238, 1.0, 0.9558981301812443, 0.27273014885581986, 0.6220492900301845, 0.8910544129978384, 0.5125720648338308, 0.948286671368632, 0.7487484405394067, 0.6654756349251802, 0.24475658565415942, 0.0, 0.35953246565880087, 0.09515288759613828, 0.6333966919634638, 0.6652886245906603],
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
        budget = 1200
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
        
        update_task_config_parameters(0,False, task_info["waypoint_name"])
        for train_seed_sublist in [[s] for s in train_seed_list]:
            for pars_name, x  in selected_designs.items():
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
        file_path = "cache/airframes_animationdata/1272882117_215_3_circle_airframeanimationdata.wb" 
        get_hover_policy(file_path)

    elif sys.argv[1] == "--plot-rank-error":
        from plot_src import *
        plot_accuracy_loss_vs_training_time("f_2_observed.txt")

    elif sys.argv[1] == "--evaluate-one":
        from problem_airframes import *
                # # # # Best solution
        # x = np.array([0.38186925277113914, 0.8577162381261587, 0.5892557688057423, 0.48997870832681656, 0.7854188643395901, 0.14301882404834032, 0.2946389252319932, 0.7892415830865502, 0.7922039292752743, 0.392503266222775, 0.5793885868042707, 0.5937053170055151, 0.2501097805798054, 0.8285091752186418, 0.09945700597018003, 0.420531983487308, 0.21779430285096169, 0.016970542259514332])

        # # # hex       
        x = np.array([
                    0.0, 0.5, 0.5000, 0.5, 0.5, 
                    0.0, 0.5, 0.1667, 0.5, 0.5, 
                    0.0, 0.5, 0.8333, 0.5, 0.5,
                    0.4, 0.4, 0.4,
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
        motor_rl_objective_function(pars, seed_train, seed_enjoy, 4000, task_info["waypoint_name"], "problem_airframes_train_and_enjoy.csv", render="visualize")
        exit(0)

    elif sys.argv[1] == "--enjoy-one":
        from problem_airframes import *
        #cache/airframes_animationdata/2614932373_1015_42_offsetcone_best_speed_airframeanimationdata.wb
        #cache/airframes_animationdata/2614932373_1015_42_offsetcone_best_efficiency_airframeanimationdata.wb
        file_path = "cache/airframes_animationdata/1978095366_1012_42_circle_best_speed_airframeanimationdata.wb" 
        animation_data  = load_animation_data_and_policy(file_path) # load policy into correct path
        save_robot_pars_to_file(animation_data["pars"])
        plot_airframe_to_file_isaacgym(animation_data["pars"], filepath="test_airframe_render.png")
        motor_position_enjoy(3, animation_data["policy_path"], task_info["waypoint_name"], "position_setpoint_task", "visualize")


    elif sys.argv[1] == "--ax-get-conclusions-solution-space":
        import plot_src
        from interfaces import problem_analyzer
        pa = problem_analyzer(f"cache/ax_optimization_status/{task_info['waypoint_name']}_6.json")

        exit(0)
        interpolated_x, was_evaluated, relative_pos_on_pareto = pa.get_pareto_solutions_with_extra_interpolated_solutions()
        plot_src.animate_solution_interpolation(interpolated_x, was_evaluated, relative_pos_on_pareto)

    else:
        print("sys.argv[1]=",sys.argv[1],"not recognized.", sep=" ")
