import sys
from matplotlib import pyplot as plt
import numpy as np
import test_vertiq_motor
import torch
from tqdm import tqdm as tqdm
exec(open("/home/paran/Dropbox/NTNU/aerial_gym_dev/aerial_gym_dev/utils/battery_rotor_dynamics.py").read())



# Load and process experimental data

freq = test_vertiq_motor.freq
dt = test_vertiq_motor.dt
propeller_idx = test_vertiq_motor.propeller_idx
n_repeat_experiment = test_vertiq_motor.n_repeat_experiment

def load_data(propeller_idx, freq):
    filename = f"results/data/vertiq_w_propidx={propeller_idx}_freq={freq}_nreps={n_repeat_experiment}.txt"
    with open(filename, "r") as f:
        time_set_list = eval(f.readline().strip())
        set_w_list = eval(f.readline().strip())
        time_request_list = eval(f.readline().strip())
        observed_w_list = eval(f.readline().strip())
    return time_set_list, set_w_list, time_request_list, observed_w_list

time_set_list, set_w_list, time_request_list, observed_w_list = load_data(propeller_idx, freq)


def compute_time_based_average(time_set_list, set_w_list, dt, max_dt_for_average):
    time_arrays = [np.array(time_set) for time_set in time_set_list]
    w_arrays = [np.array(set_w) for set_w in set_w_list]
    max_time = max(time_array[-1] for time_array in time_arrays)
    time_points = np.arange(0, max_time, dt)
    res = np.zeros(len(time_points))
    for i, t in enumerate(time_points):
        values = []
        for time_array, w_array in zip(time_arrays, w_arrays):
            idx = np.searchsorted(time_array, t)
            if idx > 0 and (idx == len(time_array) or abs(time_array[idx-1] - t) < abs(time_array[idx] - t)):
                idx -= 1
            if abs(time_array[idx] - t) <= max_dt_for_average:
                values.append(w_array[idx])
        if values:
            res[i] = np.median(values)
    return time_points, res.tolist()


def get_error_modeled_vs_observed(observed_w_list, ref_w_list, dt, motor_constant):
    rotor_dinamycs = BatteryRotorDynamics(1, 1, [propeller_idx], 8, 10.0, dt, 0.1, "cpu")
    for i in range(1000):
        rotor_dinamycs.set_desired_rps_and_get_current_rps(0.0, 0.08)

    modeled_w = []
    for w in ref_w_list:
        current_w = rotor_dinamycs.set_desired_rps_and_get_current_rps(w, motor_constant)
        modeled_w.append(current_w)
    return np.quantile(np.abs(np.array(observed_w_list)-np.array(modeled_w)), 0.90)


t_list, w_list = compute_time_based_average(time_request_list, observed_w_list, dt, max_dt_for_average=1/500)

best_error = 1e8
best_motor_constant = None
for motor_constant in tqdm(np.linspace(0.001,0.08,100)):
    error = get_error_modeled_vs_observed(w_list, set_w_list[0], dt, motor_constant)
    if error < best_error:
        best_error = error
        best_motor_constant = motor_constant
        print("motor_constant:", motor_constant, ", error:", error)


rotor_dinamycs = BatteryRotorDynamics(1, 1, [propeller_idx], 8, 10.0, dt, 0.1, "cpu")

modeled_w = []
for w in set_w_list[0]:
    current_w = rotor_dinamycs.set_desired_rps_and_get_current_rps(w, best_motor_constant)
    modeled_w.append(current_w)

# plt.plot(time_set_list[0], modeled_w, alpha=0.5)
# plt.plot(time_set_list[0], set_w_list[0], alpha=0.5)
plt.plot(time_set_list[0], modeled_w, alpha=0.5, label="modeled")
plt.plot(t_list, w_list, linestyle="", marker=".", color="red", alpha=0.5, label="observed")
plt.legend()
plt.show()
exit(0)

# for i in range(n_repeat_experiment):
#     plt.plot(time_request_list[i], observed_w_list[i], linestyle="", marker=".",)








exit(0)
avg_target_w = process_data(target_w_list, n_repeat_experiment)
avg_w_get_request_time = process_data(observed_w_get_request_time, n_repeat_experiment)

avg_times = process_time(observed_times, n_repeat_experiment)
corrected_target = corrected_target_w(avg_target_w, avg_w_get_request_time)


# Commanded w
avg_target_w

# Target corrected as the vertiq controller does not converge in exactly commanded velocity
corrected_target

# Observed w (averaged over many repetitions)
avg_w_get_request_time

# Difference between target and observed
error_wrt_target = np.abs(corrected_target - avg_w_get_request_time)
error_wrt_target[error_wrt_target < 2] = 0.0



plt.figure(figsize=(10, 6))
plt.plot(avg_times, avg_target_w, label="Commanded w")
plt.plot(avg_times, avg_w_get_request_time, label="Observed w")

plt.xlabel("Time (s)")
plt.ylabel("w")
plt.title(f"Averaged Data over {n_repeat_experiment} Repetitions")
plt.legend()
plt.grid(True)
plt.show()