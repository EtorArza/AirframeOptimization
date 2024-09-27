from matplotlib import pyplot as plt
import numpy as np
import test_vertiq_motor

freq = test_vertiq_motor.freq
dt = test_vertiq_motor.dt
propeller_idx = test_vertiq_motor.propeller_idx
n_repeat_experiment = test_vertiq_motor.n_repeat_experiment

def load_data(propeller_idx, freq):
    filename = f"results/data/vertiq_w_propidx={propeller_idx}_freq={freq}_nreps={n_repeat_experiment}.txt"
    with open(filename, "r") as f:
        observed_times = np.array(eval(f.readline().strip()))
        target_w_list = np.array(eval(f.readline().strip()))
        observed_w_read_time = np.array(eval(f.readline().strip()))
        observed_w_get_request_time = np.array(eval(f.readline().strip()))
    return observed_times, target_w_list, observed_w_read_time, observed_w_get_request_time

def process_data(data, n_repeat_experiment):
    single_exp_length = len(data) // n_repeat_experiment
    reshaped_data = data.reshape(n_repeat_experiment, single_exp_length)
    averaged_data = np.mean(reshaped_data, axis=0)
    return averaged_data

def process_time(observed_times, n_repeat_experiment):
    single_exp_length = len(observed_times) // n_repeat_experiment
    reshaped_times = observed_times.reshape(n_repeat_experiment, single_exp_length)
    adjusted_times = reshaped_times - reshaped_times[:, 0][:, np.newaxis]
    averaged_times = np.mean(adjusted_times, axis=0)
    return averaged_times

def corrected_target_w(avg_target_w, avg_w_read_time):
    n_levels_website_data = len(test_vertiq_motor.vertiq_w.tolist())
    change_indexes = [0] + [i for i in range(1, len(avg_target_w)) if avg_target_w[i] != avg_target_w[i-1]] + [-1]
    res = np.zeros_like(avg_target_w)
    for i in range(len(change_indexes)-1):
        res[change_indexes[i]:change_indexes[i+1]] = np.mean(avg_w_read_time[int(change_indexes[i]+0.25*freq):change_indexes[i+1]])
    return res
 
observed_times, target_w_list, observed_w_read_time, observed_w_get_request_time = load_data(propeller_idx, freq)

avg_target_w = process_data(target_w_list, n_repeat_experiment)
avg_w_read_time = process_data(observed_w_read_time, n_repeat_experiment)
avg_w_get_request_time = process_data(observed_w_get_request_time, n_repeat_experiment)

avg_times = process_time(observed_times, n_repeat_experiment)
corrected_target = corrected_target_w(avg_target_w, avg_w_read_time)


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