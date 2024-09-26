from matplotlib import pyplot as plt

freq = 500
dt = 1.0 / freq
propeller_idx = 0
n_repeat_experiment = 1


def load_data(propeller_idx, freq):
    filename = f"results/data/vertiq_w_propidx={propeller_idx}_freq={freq}_nreps={n_repeat_experiment}.txt"
    
    with open(filename, "r") as f:
        observed_times = eval(f.readline().strip())
        target_w_list = eval(f.readline().strip())
        observed_w_read_time = eval(f.readline().strip())
        observed_w_get_request_time = eval(f.readline().strip())
    return observed_times,target_w_list,observed_w_read_time,observed_w_get_request_time   




observed_times,target_w_list,observed_w_read_time,observed_w_get_request_time = load_data(propeller_idx, freq)

plt.plot(observed_times, target_w_list, label="Target w")
plt.plot(observed_times, observed_w_read_time, label="w read time")
plt.plot(observed_times, observed_w_get_request_time, label="w req time")
plt.xlabel("Time")
plt.ylabel("w")
plt.legend()
plt.show()