import sys
from tqdm import tqdm as tqdm
import numpy as np
from interfaces import *


task_info = {
    "waypoint_name": "offsetcone",
    "threshold_n_waypoints_per_reset": 18.0,
    "threshold_n_waypoints_reachable_based_on_battery_use": 5.0
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



    # Plot how time per evaluation in snobfit increases linearly
    elif sys.argv[1] == "--plot-snobfit-time-per-1000-evaluations":
        from matplotlib import pyplot as plt
        a = [35.672,66.487,83.07,106.848,138.234,164.302,165.799,150.121,209.833,222.891,271.217,238.373,299.406,275.698,383.726,350.589,306.331,362.808,344.787,466.726000000001,495.216,544.442,483.626,572.485,505.259,523.818,644.181000000001,629.82,687.755999999999,601.378000000001,670.198,619.061,793.431000000001,791.849,738.907999999999,605.154999999999,705.071000000002,752.395999999999,727.92,760.445,774.154000000002,1010.754,1039.556,1055.197,1093.278,1070.267,1209.765,1240.372,1382.626,1265.622,1418.167,1332.901,1192.581,1425.083,1391.77,1585.827,1363.194,1522.193,1248.111,1561.504,1648.914,1860.214,1868.236,1732.594,1893.023,1554.103,2117.976,1658.75999999999,1469.963,1273.931]
        plt.plot(a)
        plt.ylabel("time (s)")
        plt.xlabel("x 1000 evaluations")
        plt.title("Snobfit: time per 1000 evaluations")
        plt.show()


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


    else:
        print("sys.argv[1]=",sys.argv[1],"not recognized.", sep=" ")
