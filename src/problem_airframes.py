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






def from_0_1_to_RobotParameter(x_0_1: numpy.typing.NDArray[np.float_],  motor_idx_0_1=None):

    '''
    Every value in x_0_1 is in the interval [0,1], where 0 represents lowest possible value, and 1 represents highest possible value.
    
    
             ---Position (polar)------                ------Orientation-----
    x_0_1 = [r_0_1, theta_0_1, phi_0_1,               eulerx_0_1, eulerz_0_1,   # propeller 1
             r_0_1, theta_0_1, phi_0_1,               eulerx_0_1, eulerz_0_1,   # propeller 2
    etc.]
    '''
    battery_S = 4

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

        motor_orientations[i][0] = scale_from_0_1(x_0_1[i*5+3], -90.0 if i==0 else -60.0, 90.0 if i==0 else 60.0)
        motor_orientations[i][1] = 0.0
        motor_orientations[i][2] = scale_from_0_1(x_0_1[i*5+4], 0.0, 360.0)

        motor_orientations[i + n_motors//2][0] = -motor_orientations[i][0]
        motor_orientations[i + n_motors//2][1] = 0.0
        motor_orientations[i + n_motors//2][2] = 360.0 - motor_orientations[i][2]

    pars = RobotParameter()
    pars.n_motors = n_motors
    assert n_motors in (4,6)
    pars.motor_directions = [1,-1,-1,-1,1,1] if pars.n_motors == 6 else [1,-1,-1, 1] if pars.n_motors == 4 else None 

    pars.motor_translations = motor_translations.tolist()
    pars.motor_orientations = motor_orientations.tolist()

    from aerial_gym_dev.utils.battery_rotor_dynamics import BatteryRotorDynamics
    compatible_motors, compatible_batteries = BatteryRotorDynamics.get_compatible_battery_and_motors_indices(battery_S)
    n_compatible_motors, _ = len(compatible_motors), len(compatible_batteries)

    pars.motor_idx_list = [compatible_motors[int(el*(n_compatible_motors-1e-8))]  for el in motor_idx_0_1]
    pars.motor_idx_list = [*pars.motor_idx_list, *pars.motor_idx_list] # Same motors on the other side to keep simmetry
    pars.prop_diameters = []
    pars.battery_idx = 3 #compatible_batteries[int(battery_idx_0_1*(n_compatible_batteries-1e-8))]




    motor_only_masses, pars.battery_mass, prop_diameters_list = BatteryRotorDynamics.get_motor_and_battery_mass(pars.motor_idx_list, pars.battery_idx)


    guard_and_arm_mass = 0.008
    pars.motor_masses = [mtr_mass + guard_and_arm_mass for mtr_mass in motor_only_masses] # actually arms mass

    pars.base_mass = 0.3
    pars.base_inertia_matrix = np.array([
      [	 6.831e-04,  -1.000e-07, -2.100e-06],
      [ -1.000e-07,   7.673e-04, -3.700e-06],
      [ -2.100e-06,  -3.700e-06,  6.401e-04],
    ]) 
    pars.prop_diameters = prop_diameters_list[:]

    pars.total_mass = sum(pars.motor_masses) + pars.base_mass




    #pars.sensor_masses = [0.15, 0.1]
    #pars.sensor_orientations = [[0,0,0],[0,-20,0]]
    #pars.sensor_translations = [[0,0.5,0.1],[0,0,-0.1]]
    pars.cq = 0.1

    pars = repair_pars_fabrication_constraints(pars)

    return pars


def f_symmetric_hexarotor_0_1(x: numpy.typing.NDArray[np.float_], seed_train: int, seed_enjoy, task_info):
    if x.shape == (18,):
        x_0_1 = x[:15]
        motor_idx_0_1 = x[15:18]
    elif x.shape == (12,):
        x_0_1 = x[:10]
        motor_idx_0_1 = x[10:12]
    else:
        raise ValueError("Wrong dimension on x.", f"x = {x}")

    pars = from_0_1_to_RobotParameter(x_0_1, motor_idx_0_1)
    n_waypoints_per_reset, n_waypoints_reachable_based_on_battery_use = motor_rl_objective_function(pars, seed_train, seed_enjoy, 4000, task_info['waypoint_name'], f"results/data/local_solve_{task_info['waypoint_name']}.csv", "headless")
    return n_waypoints_per_reset, n_waypoints_reachable_based_on_battery_use



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


