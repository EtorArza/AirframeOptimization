import sys
import numpy as np
from tqdm import tqdm as tqdm
import time
from matplotlib import pyplot as plt
import re
exec(open("/home/paran/Dropbox/NTNU/aerial_gym_dev/aerial_gym_dev/utils/battery_rotor_dynamics.py").read())

propeller_idx = 0
test_exp = False

vertiq_w = np.array([
[0,    2105,  4244,  6378,  8466, 10495, 12461, 14376, 16195, 17970, 19672], # APC5x4R
[0,    2109,  4263,  6421,  8549, 10663, 12713, 14702, 16633, 18525, 20290, 21961, 23596, 25105, 26619],
[0,    2091,  4212,  6340,  8396, 10421, 12333, 14215, 15989, 17644, 19229],
[0,    2103,  4239,  6340,  8425, 10464, 12426, 14333, 16183, 17966, 19674, 21276, 22852],
[0,    2083,  4170,  6248,  8212, 10147, 11958, 13732, 15395],
][propeller_idx], dtype=np.float64) * np.pi / 30

vertiq_tune_pars = [
    # {"velocity_ff0":0.0, "velocity_ff1": 0.0043, "velocity_ff2": 0.000000484, # manufacturer values
    {"velocity_ff0":0.0, "velocity_ff1": 0.00433, "velocity_ff2": 2.002e-07, "velocity_Kp":0.035, "velocity_Ki":0.0015, "velocity_Kd":0.0004},
][propeller_idx]


def run_in_real_motor(target_w_list, freq, save_cycle_max_offset, n_repeat_experiment, resfile_prefix):
    import iqmotion as iq
    dt = 1.0 / freq
    com = iq.SerialCommunicator("/dev/ttyUSB0", baudrate=921600)
    module = iq.Vertiq2306(com, firmware="speed")
    module.set("propeller_motor_control", "timeout", 0.5)
    module.set("propeller_motor_control", "ctrl_coast")
    for key, value in vertiq_tune_pars.items():
        module.set("propeller_motor_control", key, value)

    time_set_list = []
    set_w_list = []
    time_request_list = []
    observed_w_list = []
    for k in tqdm(range(n_repeat_experiment)):
        waiting_w_read = False
        time_set_list.append([])
        set_w_list.append([])
        time_request_list.append([])
        observed_w_list.append([])
        start = time.time()
        time_request = time.time()
        save_reminder = int((k/n_repeat_experiment)*save_cycle_max_offset) # 0 1 2 ... 11
        for i, target_w in enumerate(target_w_list):
            if not waiting_w_read and i%save_cycle_max_offset==save_reminder:
                module.get_async("brushless_drive", "obs_velocity")
                time_request = time.time() - start
                waiting_w_read = True

            while time.time() - start < i/freq:
                continue

            module.set("propeller_motor_control", "ctrl_velocity", target_w)
            time_set_list[-1].append(time.time()-start)
            set_w_list[-1].append(target_w)

            while time.time() - start < (i+1)/freq:
                if waiting_w_read:
                    module.update_replies()
                    if module.is_fresh("brushless_drive", "obs_velocity"):
                        reply = module.get_reply("brushless_drive", "obs_velocity")
                        time_request_list[-1].append(time_request)
                        observed_w_list[-1].append(reply)
                        waiting_w_read = False
                else:
                    time.sleep(1e-5)
        while waiting_w_read:
            module.update_replies()
            if module.is_fresh("brushless_drive", "obs_velocity"):
                reply = module.get_reply("brushless_drive", "obs_velocity")
                time_request_list[-1].append(time_request)
                observed_w_list[-1].append(reply)
                waiting_w_read = False
        module.set("propeller_motor_control", "ctrl_coast")
        with open(f"scripts/vertiq_motor_modeling/run_results/calibration_propidx={propeller_idx}_freq={freq}_nreps={n_repeat_experiment}.txt", "w") as f:
            print(time_set_list, file=f)
            print(set_w_list, file=f)
            print(time_request_list, file=f)
            print(observed_w_list, file=f)
        print("Cooling pause...")
        time.sleep(30)



    print("Time:",time.time() - start, "s")

    #Set the module to coast, then wait 2 seconds
    module.set("propeller_motor_control", "ctrl_coast")
    time.sleep(0.2)


    print("done", propeller_idx, freq)

def read_real_motor_data_into_average(filename):
    freq = int(re.search(r'freq=(\d+)', filename).group(1))
    dt = 1/freq
    with open(filename, "r") as f:
            time_set_list = eval(f.readline().strip())
    _time_arrays = [np.array(time_set) for time_set in time_set_list]
    _max_time = max(time_array[-1] for time_array in _time_arrays)
    t_list = np.arange(0, _max_time, dt)
    
    with open(filename, "r") as f:
        time_set_list_of_lists = eval(f.readline().strip())
        set_w_list_of_lists = eval(f.readline().strip())
        time_request_list_of_lists = eval(f.readline().strip())
        observed_w_list_of_lists = eval(f.readline().strip())

    def compute_time_based_average(t_list, observed_t_list_of_lists, w_list_of_lists, dt, max_dt_for_average):
        time_arrays = [np.array(time_set) for time_set in observed_t_list_of_lists]
        w_arrays = [np.array(set_w) for set_w in w_list_of_lists]
        
        res = np.zeros(len(t_list))
        for i, t in enumerate(t_list):
            values = []
            for time_array, w_array in zip(time_arrays, w_arrays):
                idx = np.searchsorted(time_array, t)
                if idx > 0 and (idx == len(time_array) or abs(time_array[idx-1] - t) < abs(time_array[idx] - t)):
                    idx -= 1
                if abs(time_array[idx] - t) <= max_dt_for_average:
                    values.append(w_array[idx])
            if values:
                res[i] = np.median(values)
        return res.tolist()

    w_list_set = compute_time_based_average(t_list, time_set_list_of_lists, set_w_list_of_lists, dt, max_dt_for_average=1/freq)
    w_list_observed = compute_time_based_average(t_list, time_request_list_of_lists, observed_w_list_of_lists, dt, max_dt_for_average=1/freq)
    return t_list, w_list_set, w_list_observed


if __name__ == "__main__":

    # Run calibration procedure on real hardware
    if sys.argv[1] == "--run-calibration":
        import random
        freq = 500
        time.sleep(2)

        w_00 = vertiq_w[1]
        w_15 = w_00 + int(0.15 * (vertiq_w[-1] - w_00))
        w_25 = w_00 + int(0.25 * (vertiq_w[-1] - w_00))
        w_35 = w_00 + int(0.35 * (vertiq_w[-1] - w_00))
        w_50 = w_00 + int(0.50 * (vertiq_w[-1] - w_00))
        w_75 = w_00 + int(0.75 * (vertiq_w[-1] - w_00))
        w_max = vertiq_w[-1]

        w_discrete = [w_00, w_15, w_25, w_35, w_50, w_75, w_max]


        target_w_list = []

        if test_exp: # smaller dataset to finish fast
            for target_w in vertiq_w[:-6]:
                target_w_list += (freq // 2) * [target_w.item()]
            target_w_list+= (freq // 2) * [vertiq_w[0]]
        else:
            # Go over all values
            for target_w in vertiq_w:
                target_w_list += (freq // 2) * [target_w.item()]

            # Go from min to every value and back.
            target_w_list += (freq // 2) * [w_discrete[0]]
            for target_w in w_discrete[1:]:
                target_w_list += (freq // 2) * [w_discrete[0]]
                target_w_list += (freq // 2) * [target_w]
            target_w_list += (freq // 2) * [w_discrete[0]]

        run_in_real_motor(target_w, freq, save_cycle_max_offset=10, n_repeat_experiment=40, resfile_prefix="calibration")

    if sys.argv[1] == "--plot-calibration":
        import re
        filename = f"scripts/vertiq_motor_modeling/run_results/calibration_propidx=0_freq=500_nreps=40.txt"
        propidx = int(re.search(r'propidx=(\d+)', filename).group(1))
        freq = int(re.search(r'freq=(\d+)', filename).group(1))
        nreps = int(re.search(r'nreps=(\d+)', filename).group(1))
        dt = 1/freq

        t_list, w_list_set, w_list_observed = read_real_motor_data_into_average(filename)

        def get_error_modeled_vs_observed(w_list_observed, w_list_set, dt, motor_constant):
            rotor_dinamycs = BatteryRotorDynamics(1, 1, [propeller_idx], 8, 10.0, dt, 0.1, "cpu")
            for i in range(1000):
                rotor_dinamycs.set_desired_rps_and_get_current_rps(0.0, 0.08)

            modeled_w = []
            for w in w_list_set:
                current_w = rotor_dinamycs.set_desired_rps_and_get_current_rps(w, motor_constant)
                modeled_w.append(current_w)
            return np.quantile(np.abs(np.array(w_list_observed)-np.array(modeled_w)), 0.90)



        best_error = 1e8
        best_motor_constant = None
        for motor_constant in tqdm(np.linspace(0.001,0.08,500)):
            error = get_error_modeled_vs_observed(w_list_observed, w_list_set, dt, motor_constant)
            if error < best_error:
                best_error = error
                best_motor_constant = motor_constant
                print("motor_constant:", motor_constant, ", error:", error)
 

        rotor_dinamycs = BatteryRotorDynamics(1, 1, [propeller_idx], 8, 10.0, dt, 0.1, "cpu")

        modeled_w = []
        for w in w_list_set:
            current_w = rotor_dinamycs.set_desired_rps_and_get_current_rps(w, best_motor_constant)
            modeled_w.append(current_w)

        # plt.plot(time_set_list[0], modeled_w, alpha=0.5)
        # plt.plot(time_set_list[0], set_w_list[0], alpha=0.5)
        plt.plot(t_list, modeled_w, alpha=0.5, label="modeled")
        plt.plot(t_list, w_list_observed, linestyle="", marker=".", color="red", alpha=0.5, label="observed")
        plt.legend()
        plt.show()
        exit(0)

        # for i in range(n_repeat_experiment):
        #     plt.plot(time_request_list[i], observed_w_list[i], linestyle="", marker=".",)


    if sys.argv[1] == "--run-rl-agent-commands-in-real-motor":
        from matplotlib import pyplot as plt

        with open('desired_w_rl_agent.txt', 'r') as f:
            target_w_list = [float(line.strip()) for line in f]
        target_w_list = target_w_list
        run_in_real_motor(target_w_list, freq=100, save_cycle_max_offset=2, n_repeat_experiment=4, resfile_prefix="rlactions")



    if sys.argv[1] == "--plot-rl-agent-commands":
        from matplotlib import pyplot as plt
        filename = "scripts/vertiq_motor_modeling/run_results/calibration_propidx=0_freq=100_nreps=4.txt"
        propidx = int(re.search(r'propidx=(\d+)', filename).group(1))
        freq = int(re.search(r'freq=(\d+)', filename).group(1))
        nreps = int(re.search(r'nreps=(\d+)', filename).group(1))
        dt = 1/freq

        t_list, w_list_set, w_list_observed = read_real_motor_data_into_average(filename)

        rotor_dinamycs = BatteryRotorDynamics(1, 1, [propeller_idx], 8, 10.0, dt, 0.1, "cpu")
        for i in range(100):
            rotor_dinamycs.set_desired_rps_and_get_current_rps(0.0, 0.00001)

        modeled_w = []
        for w in w_list_set:
            current_w = rotor_dinamycs.set_desired_rps_and_get_current_rps(w, 0.0186)
            modeled_w.append(current_w)

        plt.plot(t_list, w_list_set, label="rl-commands")
        plt.plot(t_list, modeled_w, label="model")
        plt.plot(t_list, w_list_observed, linestyle="", marker=".", color="red", alpha=0.5, label="observed")
        plt.legend()
        plt.show()