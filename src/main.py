import sys
from tqdm import tqdm as tqdm
import numpy as np


task_info = {
    "waypoint_name": "offsetcone",
    "threshold_n_waypoints_per_reset": 8.0,
    "threshold_n_waypoints_reachable_based_on_battery_use": 200.0
}

selected_designs = {
    "baseline_2_2_2_motors":[0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5, 0.37114358066875286, 0.37114358066875286, 0.37114358066875286],
    "fast":[0.07715437527428756, 0.6218957229149574, 0.49765854154815503, 0.7309625753736333, 0.5001311977397834, 0.43157054869476463, 0.6387684240251704, 0.7630872292384648, 0.9600682206027545, 0.6958757520141683, 0.2513954749425118, 0.46514910716525726, 0.21839277831997495, 0.02373766483805347, 0.21094738572938934, 0.24814489622916597, 0.37114358066875286, 0.2870870028106025],
    "efficient":[0.07285065297852857, 0.6129069471982204, 0.5385502373788117, 0.704581536252464, 0.5340970073893634, 0.3084352243345762, 0.6654766925521296, 0.7102241220675533, 1.0, 0.6397600507091973, 0.1968733075763761, 0.5507576315448027, 0.1972728073004909, 0.0, 0.1920740597049365, 0.2530078418283584, 0.440390566598894, 0.3101391595022589],
    "baseline_2_2_4_motors":[0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5, 0.2530078418283584, 0.440390566598894, 0.3101391595022589],
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
        num_rpms = component_data.rpms.shape[1]

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
        from interfaces import *
        import itertools

        train_seed_list = list(range(2,23))
        enjoy_seed_list = list(range(42,44))
        max_epochs = 4000
        
        for pars_name, x  in selected_designs.items():

            x = np.array(x)
            x_0_1 = x[:15]
            motor_idx_0_1 = x[15:18]
        
            pars = from_0_1_to_RobotParameter(x_0_1, motor_idx_0_1)
            print(pars)
            airframe_repeatedly_train_and_enjoy(train_seed_list, enjoy_seed_list, max_epochs, pars, task_info, f"results/data/repeatedly_train_chosen_designs_{task_info['waypoint_name']}.csv")






    elif sys.argv[1] == "--airframes-f-variance-plot":
        import plot_src
        plot_src.boxplots_repeatedly_different_train_seed("results/data/repeatedly_train_chosen_designs_offsetcone.csv", "offsetcone")
        exit(0)
        plot_src.generate_bokeh_interactive_plot(f"results/data/local_solve_offsetcone.csv", "offsetcone")
        # plot_src.boxplots_repeatedly_different_train_seed(f"results/data/repeatedly_standard_hex_different_train_seed_{task_info['waypoint_name']}.csv", task_info['waypoint_name'])
        # plot_src.multiobjective_scatter_by_train_time(f"results/data/details_every_evaluation_{waypoint_name}.csv")
        exit(0)

    elif sys.argv[1] == "--ax-get-conclusions-solution-space":
        import plot_src
        pa = problem_analyzer("cache/ax_optimization_status/offsetcone_6.json")
        interpolated_x, was_evaluated, relative_pos_on_pareto = pa.get_pareto_solutions_with_extra_interpolated_solutions()
        plot_src.animate_solution_interpolation(interpolated_x, was_evaluated, relative_pos_on_pareto)

    else:
        print("sys.argv[1]=",sys.argv[1],"not recognized.", sep=" ")
