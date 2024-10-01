# import aerial_gym_dev
# from aerial_gym_dev.utils import battery_rotor_dynamics
import iqmotion as iq
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
import random


time.sleep(2)

freq = 500
save_cycle_max_offset = 10
dt = 1.0 / freq
propeller_idx = 0
n_repeat_experiment = 40
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

    # target_w_list += (freq // 2) * [w_discrete[-1]]
    # for target_w in w_discrete[:-1]:
    #     target_w_list += (freq // 2) * [w_discrete[-1]]
    #     target_w_list += (freq // 2) * [target_w]
    # target_w_list += (freq // 2) * [w_discrete[-1]]
    # target_w_list += (freq // 2) * [w_discrete[0]]


if __name__ == "__main__":

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
        with open(f"results/data/vertiq_w_propidx={propeller_idx}_freq={freq}_nreps={n_repeat_experiment}.txt", "w") as f:
            print(time_set_list, file=f)
            print(set_w_list, file=f)
            print(time_request_list, file=f)
            print(observed_w_list, file=f)
        print("Cooling pause...")
        time.sleep(60)



    print("Time:",time.time() - start, "s")

    #Set the module to coast, then wait 2 seconds
    module.set("propeller_motor_control", "ctrl_coast")
    time.sleep(0.2)




    print("done", propeller_idx, freq)
