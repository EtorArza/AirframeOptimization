aerial_gym_dev_path="/home/paran/Dropbox/aerial_gym_dev/aerial_gym_dev"

import sys
import isaacgym
sys.path.append(aerial_gym_dev_path)
from aerial_gym_dev.envs.base import robot_model, example_configs
from aerial_gym_dev.envs import *
from aerial_gym_dev.utils import analyze_robot_config
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Patch3D, Poly3DCollection, Text3D
from scipy.spatial.transform import Rotation
from aerial_gym_dev.envs.base.robot_model import RobotParameter, RobotModel
import numpy.typing
from tqdm import tqdm as tqdm

def plot_airframe_design(x: numpy.typing.ArrayLike):

    pars = from_0_1_to_airframe(x)


    assert len(pars.motor_orientations) == len(pars.motor_translations)


    fig = plt.figure()
    xlim = ylim = zlim = [-1.2,1.2]
    ax = fig.add_subplot(projection='3d', xlim=xlim, ylim=ylim, zlim=zlim)
    ax.set_xlabel('x',size=18)
    ax.set_ylabel('y',size=18)
    ax.set_zlabel('z',size=18)
    
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])

    frame_scale = 0.26
    for i in range(len(pars.motor_orientations)):
        t_vec = pars.motor_translations[i]
        R = Rotation.from_euler("xyz", pars.motor_orientations[i], degrees=True).as_matrix()
        if pars.motor_directions[i] == -1:
            color = "m"
        elif pars.motor_directions[i] == 1:
            color = "r"
        else:
            raise ValueError("Direction should be either 1 or -1. Instead, pars.motor_directions[i]=", pars.motor_directions[i])
        ax.add_line(Line3D([0,t_vec[0]],[0,t_vec[1]],[0,t_vec[2]]))
        ax.add_line(Line3D([t_vec[0],t_vec[0]+R[0,:]@e1*frame_scale],[t_vec[1],t_vec[1]+R[1,:]@e1*frame_scale],[t_vec[2],t_vec[2]+R[2,:]@e1*frame_scale],color=color)) #x
        ax.add_line(Line3D([t_vec[0],t_vec[0]+R[0,:]@e2*frame_scale],[t_vec[1],t_vec[1]+R[1,:]@e2*frame_scale],[t_vec[2],t_vec[2]+R[2,:]@e2*frame_scale],color=color)) #y
        ax.add_line(Line3D([t_vec[0],t_vec[0]-R[0,:]@e1*frame_scale],[t_vec[1],t_vec[1]-R[1,:]@e1*frame_scale],[t_vec[2],t_vec[2]-R[2,:]@e1*frame_scale],color=color)) #x
        ax.add_line(Line3D([t_vec[0],t_vec[0]-R[0,:]@e2*frame_scale],[t_vec[1],t_vec[1]-R[1,:]@e2*frame_scale],[t_vec[2],t_vec[2]-R[2,:]@e2*frame_scale],color=color)) #y
        ax.add_line(Line3D([t_vec[0],t_vec[0]+R[0,:]@e3*frame_scale],[t_vec[1],t_vec[1]+R[1,:]@e3*frame_scale],[t_vec[2],t_vec[2]+R[2,:]@e3*frame_scale],color='g')) #z

    plt.show()


def from_0_1_to_airframe(x: numpy.typing.ArrayLike):

    def _matrix_to_list_of_list(m):
        return [el for el in [row for row in m]]

    # A linear scalling to [0,1]^number_of_parameters
    l = len(x)
    assert l%6 == 0
    n_rotors = l // 6
    pars = RobotParameter

    
    pars.cq = 0.1
    pars.frame_mass = 0.5
    
    pars.motor_directions = [1, -1]*(n_rotors//2) + [1]*(n_rotors%2)
    pars.motor_masses = [0.1] * n_rotors


    pars.motor_translations = x[:(l//2)].reshape(-1,3) * 2.0 - 1.0
    pars.motor_orientations = x[(l//2):].reshape(-1, 3) * 360

    pars.motor_translations = _matrix_to_list_of_list(pars.motor_translations)
    pars.motor_orientations = _matrix_to_list_of_list(pars.motor_orientations)
    
    #pars.sensor_masses = [0.15, 0.1]
    #pars.sensor_orientations = [[0,0,0],[0,-20,0]]
    #pars.sensor_translations = [[0,0.5,0.1],[0,0,-0.1]]
    
    pars.max_u = 20
    pars.min_u = 0
    return pars


def constraint_check(x: numpy.typing.ArrayLike):
    pars = from_0_1_to_airframe(x)
    robot = RobotModel(pars)
    check1, check2 = analyze_robot_config.analyze_robot_config(robot)
    return (check1, check2)

if __name__ == "__main__":


    pars = from_0_1_to_airframe(np.array([
    0,0,0,
    0.25,0,0,
    0.25,0.25,0,
    0.25,0.25,0.25,

    0,0,0,
    0.25,0,0,
    0.25,0.25,0,
    0.25,0.25,0.25,
    ]))


    n_rotors = 4
    n_montecarlo = 2000
    success = 0
    for i in range(n_montecarlo):
        pars = from_0_1_to_airframe(np.random.random(6*n_rotors))
        robot = RobotModel(pars)
        check1, check2 = analyze_robot_config.analyze_robot_config(robot)
        success += int(check1 and check2)



    print("Proportion of acceptable solutions:", success / n_montecarlo)


    # plot_airframe_design(pars)


    # # If I change these parameters, is that the full set of possible quad configurations?

    # print(pars.motor_orientations) # [[0,365] x 3] x 4
    # print(pars.motor_translations) # [[-1,1] x 3] x 4
    # print(pars.motor_directions) # {-1,1} x 4


    # # Is there a way to visualize the drones?

    # change the parameters in envs/base/generalized_aerial_robot_config.py

    # then run scripts/example_control.py

    # plot mr design function, https://github.com/WilhelmWG/Evolutionary-Algorithm-for-Multirotor-Morphology-Search/blob/main/plotting.py

    # # How do I get a feasability check? I am looking for a function with binary output.

    # analyze_robot_config.analyze_robot_config(robot)


