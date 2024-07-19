import os
from aerial_gym_dev import AERIAL_GYM_ROOT_DIR
for path in [
AERIAL_GYM_ROOT_DIR + "/aerial_gym_dev/envs/base/tmp/generalized_model",
AERIAL_GYM_ROOT_DIR + "/resources/robots/generalized_model.urdf"]:
    if os.path.exists(path):
        os.remove(path)   

import sys
import isaacgym
from aerial_gym_dev.envs import *
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Patch3D, Poly3DCollection, Text3D
from scipy.spatial.transform import Rotation
from aerial_gym_dev.utils.robot_model import RobotParameter, RobotModel
import numpy.typing
from tqdm import tqdm as tqdm
from math import sqrt
from matplotlib.animation import FuncAnimation
from airframes_objective_functions import *
import pickle
import torch
import pytorch3d.transforms as p3d_transforms
aerial_gym_dev_path=AERIAL_GYM_ROOT_DIR+"/aerial_gym_dev"
sys.path.append(aerial_gym_dev_path)
from isaacgym import gymapi
import time
import subprocess
import pickle
import glob
import re
import tempfile
import functools






def from_0_1_to_RobotParameter(x_0_1: numpy.typing.NDArray[np.float_],  motor_idx_0_1=None, battery_idx_0_1=None, battery_S=6):

    '''
    Every value in x_0_1 is in the interval [0,1], where 0 represents lowest possible value, and 1 represents highest possible value.
    
    
             ---Position (polar)------                ------Orientation-----
    x_0_1 = [r_0_1, theta_0_1, phi_0_1,               eulerx_0_1, eulerz_0_1,   # propeller 1
             r_0_1, theta_0_1, phi_0_1,               eulerx_0_1, eulerz_0_1,   # propeller 2
    etc.]
    '''

    # 5 parameters per rotor, 6 rotors in total. We only define 3 rotors, due to simmetry.
    # assert x.shape == (5*3,) or x.shape == (5*2,), "x.shape = "+ str(x.shape)

    def scale_from_0_1(x_0_1, min_value, max_value):
        return min_value + x_0_1*(max_value - min_value)


    def polar_to_cartesian(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    assert isinstance(x_0_1,np.ndarray), "x_0_1 = "+str(x)+" | type(x_0_1) = "+str(type(x_0_1))
    assert len(x_0_1) // 5 == len(motor_idx_0_1)

    n_motors = (len(x_0_1) // 5) * 2
    motor_translations = np.zeros((n_motors, 3))
    motor_orientations = np.zeros((n_motors, 3))

    for i in range(n_motors//2):
        r =     scale_from_0_1(x_0_1[i*5]  , 0.11875    , 0.25)
        theta = scale_from_0_1(x_0_1[i*5+1], 0.1*np.pi  , 0.9*np.pi)
        phi =   scale_from_0_1(x_0_1[i*5+2], 0.0        , 1.0*np.pi)
        x, y, z = polar_to_cartesian(r, theta, phi)
        motor_translations[i][0] = x
        motor_translations[i][1] = y
        motor_translations[i][2] = z

        motor_translations[i + n_motors//2][0] = x
        motor_translations[i + n_motors//2][1] = -y
        motor_translations[i + n_motors//2][2] = z

        motor_orientations[i][0] = scale_from_0_1(x_0_1[i*5+3], 360*0.4, 360*0.6)
        motor_orientations[i][1] = 0.0
        motor_orientations[i][2] = scale_from_0_1(x_0_1[i*5+3], 0.0, 360.0)

        motor_orientations[i + n_motors//2][0] = scale_from_0_1(1.0 - x_0_1[i*5+3], 360*0.4, 360*0.6)
        motor_orientations[i + n_motors//2][1] = 0.0
        motor_orientations[i + n_motors//2][2] = scale_from_0_1(1.0 - x_0_1[i*5+3], 0.0, 360.0)

    pars = RobotParameter()
    pars.n_motors = n_motors
    assert n_motors in (4,6)
    pars.motor_directions = [1,-1,-1,-1,1,1] if pars.n_motors == 6 else [1,-1,-1, 1] if pars.n_motors == 4 else None 

    pars.motor_translations = motor_translations.tolist()
    pars.motor_orientations = motor_orientations.tolist()

    from aerial_gym_dev.utils.battery_rotor_dynamics import BatteryRotorDynamics
    compatible_motors, compatible_batteries = BatteryRotorDynamics.get_compatible_battery_and_motors_indices(battery_S)
    n_compatible_motors, n_compatible_batteries = len(compatible_motors), len(compatible_batteries)

    pars.motor_idx_list = [compatible_motors[int(el*(n_compatible_motors-1e-8))]  for el in motor_idx_0_1]
    pars.motor_idx_list = [*pars.motor_idx_list, *pars.motor_idx_list] # Same motors on the other side to keep simmetry
    pars.battery_idx = compatible_batteries[int(battery_idx_0_1*(n_compatible_batteries-1e-8))]

                     # core mass       # electronics mass
    pars.core_mass = 0.090       +     0.031
    guard_and_arm_mass = 0.008


    motor_only_masses, pars.battery_mass = BatteryRotorDynamics.get_motor_and_battery_mass(pars.motor_idx_list, pars.battery_idx)


    # For Inertia only these two below are taken into account
    pars.motor_masses = [mtr_mass + guard_and_arm_mass for mtr_mass in motor_only_masses] # actually arms mass
    pars.frame_mass = pars.core_mass + pars.battery_mass


    pars.total_mass = sum(pars.motor_masses) + pars.frame_mass
    print("------")
    print("Thrust to weight ratio: ", BatteryRotorDynamics.static_get_thrust_to_weight_ratio(pars.motor_idx_list, pars.total_mass))
    print("motor_masses =", pars.motor_masses)
    print("frame_mass =", pars.frame_mass)
    print("total_mass =",pars.total_mass)
    print("------")


    #pars.sensor_masses = [0.15, 0.1]
    #pars.sensor_orientations = [[0,0,0],[0,-20,0]]
    #pars.sensor_translations = [[0,0.5,0.1],[0,0,-0.1]]
    
    pars.motor_time_constant_min = 0.027
    pars.motor_time_constant_max = 0.037
    pars.cq = 0.1

    pars = repair_pars_fabrication_constraints(pars)

    return pars


def f_symmetric_hexarotor_0_1(x: numpy.typing.NDArray[np.float_], seed_train: int, seed_enjoy, task_info):
    if x.shape == (19,):
        x_0_1 = x[:15]
        motor_idx_0_1 = x[15:18]
        battery_idx_0_1 = x[18]
    elif x.shape == (13,):
        x_0_1 = x[:10]
        motor_idx_0_1 = x[10:12]
        battery_idx_0_1 = x[12]
    else:
        raise ValueError("Wrong dimension on x.", f"x = {x}")

    pars = from_0_1_to_RobotParameter(x_0_1, motor_idx_0_1, battery_idx_0_1)
    info_dict = motor_rl_objective_function(pars, seed_train, seed_enjoy, 4000, task_info['waypoint_name'], f"results/data/local_solve_{task_info['waypoint_name']}.csv")
    return info_dict



def repair_pars_fabrication_constraints(pars: RobotParameter) -> RobotParameter:
    res = check_collision_and_repair_isaacgym(pars)[0]
    assert isinstance(res,RobotParameter)
    return res


def get_cached_file(pars):
    pattern = f"cache/airframes_animationdata/{hash(pars)}_*_*_*_airframeanimationdata.wb"
    matching_files = glob.glob(pattern)

    if len(matching_files) == 1:
        file_path = matching_files[0]
        filename = os.path.basename(file_path)
        # Extract seed_train, seed_enjoy, and waypoint_name using regex
        match = re.match(r'\d+_(\d+)_(\d+)_(\w+)_airframeanimationdata\.wb', filename)
        if match:
            seed_train = int(match.group(1))
            seed_enjoy = int(match.group(2))
            waypoint_name = match.group(3)
            return file_path, seed_train, seed_enjoy, waypoint_name
        else:
            raise ValueError(f"Unable to extract values from filename: {filename}")
    elif len(matching_files) == 0:
        raise FileNotFoundError(f"No file found matching pattern: {pattern}")
    else:
        raise RuntimeError(f"Multiple files found matching pattern: {pattern}")


if __name__ == "__main__":

    train_and_enjoy = True
    if train_and_enjoy:

        # # # Best solution
        # x = np.array([0.08183876204900542, 0.5219701806265, 0.11665093239221351, 0.7378768213735014, 0.508831640622941, 0.0, 0.6974289414187024, 0.49857822070195307, 0.5439566454472291, 0.2208295263646864, 0.3719121034203965, 0.686407066588435, 0.8705630953168567, 0.4697827032516831, 0.6904855778324281])

        # # hex       
        x = np.array([
                    0.0, 0.5, 0.5000, 0.5, 0.5, 
                    0.0, 0.5, 0.1667, 0.5, 0.5, 
                    0.0, 0.5, 0.8333, 0.5, 0.5,
                    ])
        
        # # quad
        # x = np.array([0.0, 0.5, 0.25, 0.5, 0.5, 
        #               0.0, 0.5, 0.75, 0.5, 0.5, 
        #              ])

        pars = from_0_1_to_RobotParameter(x, [1.0, 1.0, 1.0], 0.5)
        plot_airframe_to_file_isaacgym(pars, filepath="test_airframe_render.png")
        # plot_admisible_set(pars)

        save_robot_pars_to_file(pars)
        # plot_airframe_to_file_isaacgym(pars, filepath="demo_image.png")

        seed_train = 999
        seed_enjoy = 999
        start = time.time()
        info_dict = motor_rl_objective_function(pars, seed_train, seed_enjoy, 4000, "offsetcone", "problem_airframes_train_and_enjoy.csv")
        f = loss_function(info_dict)
        print(f"objective function (train + enjoy) time: {time.time() - start}")
        print("--------------------------")
        print("f(x) = ", f)
        print("Number of Waypoints Reached: ", info_dict['f_nWaypointsReached'].cpu().item())
        print("Number of Resets: ", info_dict['f_nResets'].cpu().item())
        print("Waypoints per Reset: ", (info_dict['f_nWaypointsReached'] / info_dict['f_nResets']).cpu().item())
        print("Energy per Waypoint: ", (info_dict['f_total_energy'] / torch.clamp(info_dict['f_nWaypointsReached'], min=1.0)).cpu().item())
        print("Total Energy: ", info_dict['f_total_energy'].cpu().item())
        print("--------------------------")
    else: 

        # file_path, seed_train, seed_enjoy, waypoint_name = get_cached_file(pars)
        file_path = "cache/airframes_animationdata/7399056118471101504_21_43_offsetcone_airframeanimationdata.wb"
        animation_data  = load_animation_data_and_policy(file_path) # load policy into correct path
        save_robot_pars_to_file(animation_data["pars"])
        plot_airframe_to_file_isaacgym(animation_data["pars"], filepath="test_airframe_render.png")
        motor_position_enjoy(animation_data["seed_enjoy"], False, animation_data["waypoint_name"])

