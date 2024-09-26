# import aerial_gym_dev
# from aerial_gym_dev.utils import battery_rotor_dynamics
import iqmotion as iq
import time
import numpy as np
from matplotlib import pyplot as plt


time.sleep(5)

freq = 500
dt = 1.0 / freq
propeller_idx = 0
n_repeat_experiment = 1


vertiq_w = np.array([
[0,    2105,  4244,  6378,  8466, 10495, 12461, 14376, 16195, 17970, 19672, 21211],
[0,    2109,  4263,  6421,  8549, 10663, 12713, 14702, 16633, 18525, 20290, 21961, 23596, 25105, 26619],
[0,    2091,  4212,  6340,  8396, 10421, 12333, 14215, 15989, 17644, 19229],
[0,    2103,  4239,  6340,  8425, 10464, 12426, 14333, 16183, 17966, 19674, 21276, 22852],
[0,    2083,  4170,  6248,  8212, 10147, 11958, 13732, 15395],
][propeller_idx], dtype=np.float64) * np.pi / 30


w_00 = vertiq_w[1]
w_25 = w_00 + int(0.25 * (vertiq_w[-1] - w_00))
w_50 = w_00 + int(0.50 * (vertiq_w[-1] - w_00))
w_75 = w_00 + int(0.75 * (vertiq_w[-1] - w_00))
w_max = vertiq_w[-1]

w_discrete = [w_00, w_25, w_50, w_75, w_max]


target_w_list = []

# Go over all values
for target_w in vertiq_w:
    target_w_list += (2*freq) * [target_w.item()]


# Go from min to every value & from max to every value.
target_w_list += (freq // 2) * [w_discrete[0]]
for target_w in w_discrete[1:]:
    target_w_list += (freq // 2) * [w_discrete[0]]
    target_w_list += (freq // 2) * [target_w]

target_w_list += (freq // 2) * [w_discrete[-1]]
for target_w in w_discrete[:-1]:
    target_w_list += (freq // 2) * [w_discrete[-1]]
    target_w_list += (freq // 2) * [target_w]
target_w_list += (freq // 2) * [w_discrete[-1]]
target_w_list += (freq // 2) * [w_discrete[0]]




com = iq.SerialCommunicator("/dev/ttyUSB0")
module = iq.SpeedModule(com)
module.set("propeller_motor_control", "timeout", 0.05)



target_w_list *= n_repeat_experiment

prev_it = 0
observed_times = [0]*len(target_w_list)
observed_w_read_time = np.array([0]*len(target_w_list))
observed_w_get_request_time = np.array([0]*len(target_w_list))

waiting_w_read = False
last_read = -1
start = time.time()
for i, target_w in enumerate(target_w_list):
    p = i / len(target_w_list)

    if not waiting_w_read:
        module.get_async("brushless_drive", "obs_velocity")
        module.update_replies()
        waiting_w_read = True

    module.set("propeller_motor_control", "ctrl_velocity", target_w)

    time_next_step = (i+1)/freq
    observed_times[i] = time.time() - start
    observed_w_read_time[i] = observed_w_read_time[max(0,i-1)] # gets updated if read is successsful
    observed_w_get_request_time[i] = observed_w_get_request_time[max(0,i-1)] # gets updated if read is successsful
    while observed_times[i] < time_next_step:
        observed_times[i] = time.time() - start
        if waiting_w_read:
            module.update_replies()
            if module.is_fresh("brushless_drive", "obs_velocity"):
                reply = module.get_reply("brushless_drive", "obs_velocity")
                observed_w_read_time[i] = reply
                observed_w_get_request_time[(last_read+1):(i+1)] = reply
                waiting_w_read = False
                last_read = i
        time.sleep(1e-5)


print("Time:",start  - time.time(), "s")

#Set the module to coast, then wait 2 seconds
module.set("propeller_motor_control", "ctrl_coast")
time.sleep(0.2)

with open(f"results/data/vertiq_w_propidx={propeller_idx}_freq={freq}_nreps={n_repeat_experiment}.txt", "w") as f:
    print(observed_times, file=f)
    print(target_w_list, file=f)
    print(observed_w_read_time.tolist(), file=f)
    print(observed_w_get_request_time.tolist(), file=f)



print("done", propeller_idx, freq)

# plt.plot(observed_times, target_w_list)
# plt.show()