import sys
from tqdm import tqdm as tqdm
import numpy as np
from interfaces import *


task_info = {
    "waypoint_name": "offsetcone",
    "threshold_n_waypoints_per_reset": 8.0,
    "threshold_n_waypoints_reachable_based_on_battery_use": 200.0
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

        from aerial_gym_dev.utils.battery_rotor_dynamics import manufacturerComponentData
        from aerial_gym_dev.utils.custom_math import linear_1d_interpolation
 
        component_data = manufacturerComponentData("cpu")
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        import torch
        import os

        # Create the results/figures/motor_properties directory if it doesn't exist
        os.makedirs('results/figures/motor_properties', exist_ok=True)

        # Initialize the component_data
        component_data = manufacturerComponentData("cpu")

        # Number of motor-propeller combos and RPM values
        num_combos = len(component_data.motor_dict)
        num_rpms = component_data.rpms.shape[1]

        # Create arrays to store the calculated values
        forces = np.zeros((num_combos, num_rpms))
        efficiencies = np.zeros((num_combos, num_rpms))

        # Calculate forces and efficiencies
        for i in range(num_combos):
            c_t = component_data.motor_dict[i]["C_T"]
            rpms = component_data.motor_dict[i]["RPM"]
            currents = component_data.currents[i, :]
            
            forces[i, :] = c_t * (rpms / 60.0)**2
            efficiencies[i, :] = forces[i, :] / currents.numpy()

        # Create the forces heatmap
        plt.figure(figsize=(4, 4))
        sns.heatmap(forces, cmap="viridis", annot=False, fmt=".2f", cbar_kws={'label': 'Force (N)'})
        plt.title("Motor-Propeller Combo Forces")
        plt.xlabel("RPM Index")
        plt.ylabel("Motor-Propeller Combo Index")
        plt.tight_layout()
        plt.savefig('results/figures/motor_properties/forces.pdf')
        plt.close()

        # Create the efficiency heatmap
        plt.figure(figsize=(4, 4))
        sns.heatmap(efficiencies, cmap="viridis", annot=False, fmt=".2f", cbar_kws={'label': 'Efficiency (N/A)'})
        plt.title("Motor-Propeller Combo Efficiencies")
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

        plt.figure(figsize=(12, 8))
        sns.heatmap(interpolated_efficiencies, cmap="viridis", annot=False, fmt=".2f",
                    cbar_kws={'label': 'Efficiency (N/A)'}, mask=np.isnan(interpolated_efficiencies))
        plt.title("Efficiency with respect to Force")
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
        from problem_airframes import loss_function, dump_animation_data_and_policy, _decode_symmetric_hexarotor_to_RobotParameter_polar
        import itertools

        hex_pars = _decode_symmetric_hexarotor_to_RobotParameter_polar(np.array(
            [0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5] 
        ))


        train_seed_list = list(range(3,23))
        enjoy_seed_list = list(range(42,44))
        pars = hex_pars
        max_epochs_list = [1000, 2000, 4000]
        
        for max_epochs in max_epochs_list:
            airframe_repeatedly_train_and_enjoy(train_seed_list, enjoy_seed_list, max_epochs, pars, task_info, f"results/data/repeatedly_standard_hex_different_train_seed_{task_info['waypoint_name']}.csv")



    elif sys.argv[1] == "--airframes-repeatedly-train":
        from problem_airframes import loss_function, dump_animation_data_and_policy, _decode_symmetric_hexarotor_to_RobotParameter_polar
        import itertools

        hex_pars = _decode_symmetric_hexarotor_to_RobotParameter_polar(np.array(
            [0.0, 0.5, 0.1667, 0.5, 0.5, 0.0, 0.5, 0.5000, 0.5, 0.5, 0.0, 0.5, 0.8333, 0.5, 0.5] 
        ))
        most_waypoints_GOAT_pars = _decode_symmetric_hexarotor_to_RobotParameter_polar(np.array(
            [0.08183876204900542, 0.5219701806265, 0.11665093239221351, 0.7378768213735014, 0.508831640622941, 0.0, 0.6974289414187024, 0.49857822070195307, 0.5439566454472291, 0.2208295263646864, 0.3719121034203965, 0.686407066588435, 0.8705630953168567, 0.4697827032516831, 0.6904855778324281]
        ))
        efficient_GOAT_pars = _decode_symmetric_hexarotor_to_RobotParameter_polar(np.array(
            [0.566192737657973, 0.38650732151858636, 0.04476017193935885, 0.5451959048245645, 0.6563650895986163, 0.0, 0.4549408976618331, 0.6401616189042635, 0.6630906491835392, 0.38021388305458964, 0.7267550059333063, 0.4661118903880215, 0.7995358390157647, 0.5213023826008708, 0.7348511244117153]
        ))

        train_seed_list = list(range(2,33))
        enjoy_seed_list = list(range(42,44))
        pars_list = [hex_pars, most_waypoints_GOAT_pars, efficient_GOAT_pars]
        max_epochs = 350
        
        for pars in pars_list:
            airframe_repeatedly_train_and_enjoy(train_seed_list, enjoy_seed_list, max_epochs, pars, task_info)






    elif sys.argv[1] == "--airframes-f-variance-plot":
        import plot_src
        plot_src.boxplots_repeatedly_different_train_seed(f"results/data/repeatedly_standard_hex_different_train_seed_{task_info['waypoint_name']}.csv", task_info['waypoint_name'])
        exit(0)
        plot_src.multiobjective_scatter_by_train_time(f"results/data/details_every_evaluation_{waypoint_name}.csv")
        plot_src.generate_bokeh_interactive_plot(f"results/data/details_every_evaluation_{waypoint_name}.csv", waypoint_name)
        exit(0)

    elif sys.argv[1] == "--ax-get-conclusions-solution-space":
        import plot_src
        pa = problem_analyzer("cache/ax_optimization_status/offsetcone_6.json")
        interpolated_x, was_evaluated, relative_pos_on_pareto = pa.get_pareto_solutions_with_extra_interpolated_solutions()
        plot_src.animate_solution_interpolation(interpolated_x, was_evaluated, relative_pos_on_pareto)

    else:
        print("sys.argv[1]=",sys.argv[1],"not recognized.", sep=" ")
