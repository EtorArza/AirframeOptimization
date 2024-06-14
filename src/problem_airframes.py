aerial_gym_dev_path="/home/paran/Dropbox/NTNU/aerial_gym_dev/aerial_gym_dev"

import sys
import isaacgym
sys.path.append(aerial_gym_dev_path)
from aerial_gym_dev.envs import *
from aerial_gym_dev.utils import analyze_robot_config
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Patch3D, Poly3DCollection, Text3D
from scipy.spatial.transform import Rotation
from aerial_gym_dev.utils.robot_model import RobotParameter, RobotModel
from aerial_gym_dev.utils.example_configs import PredefinedConfigParameter
import numpy.typing
from tqdm import tqdm as tqdm
from math import sqrt
from matplotlib.animation import FuncAnimation
import subprocess
from airframes_objective_functions import motor_rl_objective_function, dump_animation_info_dict, loss_function
import pickle
import torch
import pytorch3d.transforms as p3d_transforms



animation_camera = "static" # "static" or "follow_goal"
plotlims = [-1.0, 1.0]
# Quad
quad_pars = PredefinedConfigParameter('quad')

# Hex
hex_pars = PredefinedConfigParameter('hex')



def from_0_1_to_RobotParameter(x: numpy.typing.NDArray[np.float_], task_info=None):

    '''
    Every value in x is in the interval [0,1], where 0 represents lowest possible value, and 1 represents highest possible value.
    x = [
    [rx, ry, rz, theta1, theta2, theta3], # propeller 1
    [rx, ry, rz, theta1, theta2, theta3], # propeller 2
    etc.
    ]

    '''


    # A linear scalling to [0,1]^number_of_parameters
    assert type(x)==np.ndarray, "x = "+str(x)+" | type(x) = "+str(type(x))
    assert len(x.shape)==2 and x.shape[1] == 6, "x = "+str(x)
    n_motors = x.shape[0]
    pars = RobotParameter()

    
    mass = 0.250
    proportion_mass_in_body = 0.7
    max_width = 0.26 / 2.0
    
    
    pars.cq = 0.1
    pars.frame_mass = mass * proportion_mass_in_body
    pars.motor_masses = [mass * (1.0 - proportion_mass_in_body) / n_motors] * n_motors
    

    pars.motor_directions = ([1,-1,-1,1]*6)[:n_motors]

    x_motor_translations = np.array([x[rotor_idx, i] for rotor_idx in range(n_motors) for i in range(3)])
    x_motor_orientations = np.array([x[rotor_idx, i] for rotor_idx in range(n_motors) for i in range(3,6)])

    pars.motor_translations = (x_motor_translations.reshape(-1,3) * 2.0 - 1.0) * max_width
    pars.motor_orientations = x_motor_orientations.reshape(-1, 3) * 360

    pars.motor_translations = pars.motor_translations.tolist()
    pars.motor_orientations = pars.motor_orientations.tolist()
    
    #pars.sensor_masses = [0.15, 0.1]
    #pars.sensor_orientations = [[0,0,0],[0,-20,0]]
    #pars.sensor_translations = [[0,0.5,0.1],[0,0,-0.1]]

    pars.min_thrust = pars.min_u = 0
    pars.max_thrust = pars.max_u = 3*mass*9.81 / n_motors
    pars.max_thrust_rate = 100.0
    
    
    pars.motor_time_constant_min = 0.01
    pars.motor_time_constant_max = 0.03

    pars.task_info = task_info

    return pars

def constraint_check_welf(pars: RobotParameter):
    robot = RobotModel(pars)
    check1, check2 = analyze_robot_config.analyze_robot_config(robot)
    return (check1, check2)


def constraints_fabrication(pars: RobotParameter):
    pass

def _decode_symmetric_hexarotor_to_RobotParameter(x: numpy.typing.NDArray[np.float_]):

    # 5 parameters per rotor, 6 rotors in total. We only define 3 rotors, due to simmetry.
    assert x.shape == (5*3,) or x.shape == (5*2,), "x.shape = "+ str(x.shape)
    

    n_unique_rotors = round(x.shape[0]/5)
    euler_x_max_proportion = 0.2 # Maximum rotor inclination from vertical position. We dont want them to tilt more than 20%
    def scale_down_euler_x(euler_x_01):
        return euler_x_01*euler_x_max_proportion + (0.5 - euler_x_max_proportion/2.0)
    
    simmetry_plane = "y"


    x_decoded = np.zeros(shape=(2*n_unique_rotors,6), dtype=np.float64)
    # For each propeller, symmetric encoding has 5 parameters, and 0_1 encoding has 6 x 2 (one of the propellers has simmetric parameters).
    if simmetry_plane == "y":
        for prop_i in range(n_unique_rotors):
            # og
            x_decoded[prop_i*2, 0] = x[prop_i*5 +0]                          # r_x same
            x_decoded[prop_i*2, + 1] = x[prop_i*5 +1] / 2                    # r_y can only be in one side
            x_decoded[prop_i*2, + 2] = x[prop_i*5 +2]                        # r_z same
            x_decoded[prop_i*2, + 3] = scale_down_euler_x(x[prop_i*5 +3])      # euler_x same 
            x_decoded[prop_i*2, + 4] = 0.5                                   # euler_y constant (0.5)
            x_decoded[prop_i*2, + 5] = x[prop_i*5 +4]                        # euler_z same


            # symmetric
            x_decoded[prop_i*2+1, 0] = x[prop_i*5 +0]                        # r_x same
            x_decoded[prop_i*2+1, 1] = 1.0 - (x[prop_i*5 +1]/2)              # r_y inverse, and can only be in one side
            x_decoded[prop_i*2+1, 2] = x[prop_i*5 +2]                        # r_z same

            x_decoded[prop_i*2+1, 3] = scale_down_euler_x(1.0 - x[prop_i*5 +3])# euler_x inverse 
            x_decoded[prop_i*2+1, 4] = 0.5                                   # euler_y constant (0.5)
            x_decoded[prop_i*2+1, 5] = 1.0 - x[prop_i*5 +4]                  # euler_z inverse
    return from_0_1_to_RobotParameter(x_decoded)




def f_symmetric_hexarotor_0_1(x: numpy.typing.NDArray[np.float_], seed_train: int, seed_enjoy: int, task_info: dict):
    assert x.shape == (15,) or x.shape== (10,)
    pars = _decode_symmetric_hexarotor_to_RobotParameter(x)
    pars.task_info = task_info
    info_dict = motor_rl_objective_function(pars, seed_train, seed_enjoy, 720)
    return loss_function(info_dict)

def constraint_check_welf_hexarotor_0_1(x: numpy.typing.NDArray[np.float_]):
    pars = _decode_symmetric_hexarotor_to_RobotParameter(x)
    return constraint_check_welf(pars)

def plot_airframe_design(pars:RobotParameter, translation:numpy.typing.NDArray[np.float_]=np.zeros(3), rotation_matrix: numpy.typing.NDArray[np.float_]=np.eye(3,3), target=None):

    assert translation.shape==(3,)
    assert rotation_matrix.shape==(3,3)

    assert len(pars.motor_orientations) == len(pars.motor_translations)


    fig = plt.figure()
    xlim = ylim = zlim = plotlims
    ax = fig.add_subplot(projection='3d', xlim=xlim, ylim=ylim, zlim=zlim)
    ax.set_xlabel('x',size=18)
    ax.set_ylabel('y',size=18)
    ax.set_zlabel('z',size=18)
    
    _plot_airframe_into_ax(ax, pars, translation, rotation_matrix)


    plt.show()


def plot_airframe_to_file(pars:RobotParameter, filepath):
    assert len(pars.motor_orientations) == len(pars.motor_translations)
    fig = plt.figure()
    xlim = ylim = zlim = (-0.3,0.3)
    ax = fig.add_subplot(projection='3d', xlim=xlim, ylim=ylim, zlim=zlim)
    # ax.set_xlabel('x',size=5)
    # ax.set_ylabel('y',size=5)
    # ax.set_zlabel('z',size=5)
    _plot_airframe_into_ax(ax, pars,np.zeros(3), np.eye(3,3))
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def _plot_airframe_into_ax(ax, pars:RobotParameter, translation, rotation_matrix):
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])

    frame_scale_plane = 0.016
    frame_scale_normal = 0.026

    def apply_transformation(x, translation, rotation_matrix):
        assert x.shape == (3,)
        assert translation.shape == (3,)
        assert rotation_matrix.shape == (3,3)
        T = np.concatenate([rotation_matrix, translation.reshape(3,1)], axis=1)
        T = np.concatenate([T, np.array([[0,0,0,1]])])
        res = T@(np.array([x[0],x[1],x[2], 1]).T)
        return res.flatten()[0:3]

    for i in range(len(pars.motor_orientations)):
        t_vec = pars.motor_translations[i]
        R = Rotation.from_euler("xyz", pars.motor_orientations[i], degrees=True).as_matrix()
        if pars.motor_directions[i] == -1:
            color = "b"
        elif pars.motor_directions[i] == 1:
            color = "orange"
        else:
            raise ValueError("Direction should be either 1 or -1. Instead, pars.motor_directions[i]=", pars.motor_directions[i])
        line_list = [
            [[0,t_vec[0]],[0,t_vec[1]],[0,t_vec[2]]],
            [[t_vec[0],t_vec[0]+R[0,:]@e1*frame_scale_plane],[t_vec[1],t_vec[1]+R[1,:]@e1*frame_scale_plane],[t_vec[2],t_vec[2]+R[2,:]@e1*frame_scale_plane]],
            [[t_vec[0],t_vec[0]+R[0,:]@e2*frame_scale_plane],[t_vec[1],t_vec[1]+R[1,:]@e2*frame_scale_plane],[t_vec[2],t_vec[2]+R[2,:]@e2*frame_scale_plane]],
            [[t_vec[0],t_vec[0]-R[0,:]@e1*frame_scale_plane],[t_vec[1],t_vec[1]-R[1,:]@e1*frame_scale_plane],[t_vec[2],t_vec[2]-R[2,:]@e1*frame_scale_plane]],
            [[t_vec[0],t_vec[0]-R[0,:]@e2*frame_scale_plane],[t_vec[1],t_vec[1]-R[1,:]@e2*frame_scale_plane],[t_vec[2],t_vec[2]-R[2,:]@e2*frame_scale_plane]],
            [[t_vec[0],t_vec[0]+R[0,:]@e3*frame_scale_normal],[t_vec[1],t_vec[1]+R[1,:]@e3*frame_scale_normal],[t_vec[2],t_vec[2]+R[2,:]@e3*frame_scale_normal]],
            [[0,0.8],[0,0],[0,0]],
            [[0,0],[0,0.8],[0,0]],
            [[0,0],[0,0],[0,0.8]],
        ]
        color_list = ["gray"] + [color]*4 + ['g'] + ["red","green","blue"]
        for line, color in zip(line_list, color_list):
            start = np.array([line[0][0], line[1][0], line[2][0]])
            end = np.array([line[0][1], line[1][1], line[2][1]])
            start = apply_transformation(start, translation, rotation_matrix)
            end = apply_transformation(end, translation, rotation_matrix)
            ax.add_line(Line3D([start[0],end[0]],[start[1],end[1]],[start[2],end[2]], color=color, linestyle=":" if color in ["red","green","blue"] else "-"))

def _plot_airframe_design_interactive(ax, params):

    og_pars = np.array([[el for el in params.values()]])
    pars = from_0_1_to_RobotParameter(og_pars)

    assert len(pars.motor_orientations) == len(pars.motor_translations)

    xlim = ylim = zlim = [-1.2,1.2]
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)

    
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])

    frame_scale_plane = 0.16
    frame_scale_normal = 0.26

    for i in range(len(pars.motor_orientations)):
        t_vec = pars.motor_translations[i]
        R = Rotation.from_euler("xyz", pars.motor_orientations[i], degrees=True).as_matrix()
        if pars.motor_directions[i] == -1:
            color = "blue"
        elif pars.motor_directions[i] == 1:
            color = "orange"
        else:
            raise ValueError("Direction should be either 1 or -1. Instead, pars.motor_directions[i]=", pars.motor_directions[i])
        ax.add_line(Line3D([0,t_vec[0]],[0,t_vec[1]],[0,t_vec[2]]))
        ax.add_line(Line3D([t_vec[0],t_vec[0]+R[0,:]@e1*frame_scale_plane],[t_vec[1],t_vec[1]+R[1,:]@e1*frame_scale_plane],[t_vec[2],t_vec[2]+R[2,:]@e1*frame_scale_plane],color=color)) #x
        ax.add_line(Line3D([t_vec[0],t_vec[0]+R[0,:]@e2*frame_scale_plane],[t_vec[1],t_vec[1]+R[1,:]@e2*frame_scale_plane],[t_vec[2],t_vec[2]+R[2,:]@e2*frame_scale_plane],color=color)) #y
        ax.add_line(Line3D([t_vec[0],t_vec[0]-R[0,:]@e1*frame_scale_plane],[t_vec[1],t_vec[1]-R[1,:]@e1*frame_scale_plane],[t_vec[2],t_vec[2]-R[2,:]@e1*frame_scale_plane],color=color)) #x
        ax.add_line(Line3D([t_vec[0],t_vec[0]-R[0,:]@e2*frame_scale_plane],[t_vec[1],t_vec[1]-R[1,:]@e2*frame_scale_plane],[t_vec[2],t_vec[2]-R[2,:]@e2*frame_scale_plane],color=color)) #y
        ax.add_line(Line3D([t_vec[0],t_vec[0]+R[0,:]@e3*frame_scale_normal],[t_vec[1],t_vec[1]+R[1,:]@e3*frame_scale_normal],[t_vec[2],t_vec[2]+R[2,:]@e3*frame_scale_normal],color='g')) #z

def plot_airframe_interactive_single_rotor():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    print("--------------------------------------")
    print("-----Interactive Plot-----------------")
    print("The parameter 'euler y' is irrelevant.")
    print("--------------------------------------")

    params={
    'r1':0.5,
    'r2':0.5,
    'r3':0.5,
    'euler x':0.5,
    'euler y':0.5,
    'euler z':0.5,
    }

    # Create the figure and the line that we will manipulate
    fig = plt.figure()
    xlim = ylim = zlim = [-1.2,1.2]
    ax = fig.add_subplot(projection='3d', xlim=xlim, ylim=ylim, zlim=zlim)

    _plot_airframe_design_interactive(ax, params)

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.55)



    i = -1
    axes_list = []
    slider_list = []
    update_list = []


    def make_update(i):
        def update(val):
            key = list(params.keys())[i]
            ax.clear()
            params[key] = val
            _plot_airframe_design_interactive(ax, params)
            fig.canvas.draw_idle()
        return update


    for key, item in params.items():
        i += 1
        axes_list.append(fig.add_axes([0.35, 0.3*(9-i)/9, 0.55, 0.01]))
        slider_list.append(Slider(
            ax=axes_list[i],
            label=key,
            valmin=0.0,
            valmax=1.0,
            valinit=item,
        ))


        update_list.append(make_update(i))
        slider_list[-1].on_changed(update_list[-1])

    plt.tight_layout()
    plt.show()

def animate_airframe(pars:RobotParameter, pose_list, target_list):

    max_time = 24
    dt = 0.01
    desired_fps = 24
    time_between_frames = 1.0 / 24.0


    assert len(pose_list) == len(target_list)

    start_position_list = [pose_list[0]]
    for i in range(len(pose_list)-1):

        if np.all(target_list[i] == target_list[i+1]):
            start_position_list.append(start_position_list[-1])
        else:
            start_position_list.append(pose_list[i+1])

    fig = plt.figure()
    xlim = ylim = zlim = plotlims
    ax = fig.add_subplot(projection='3d', xlim=xlim, ylim=ylim, zlim=zlim)
    ax.set_xlabel('x',size=18)
    ax.set_ylabel('y',size=18)
    ax.set_zlabel('z',size=18)

    def animate(i):
        frm_idx = int(i / dt / desired_fps)
        pose = pose_list[frm_idx]
        ax.clear()
        translation = pose[0:3]
        rotation_matrix = Rotation.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
        _plot_airframe_into_ax(ax, pars, translation, rotation_matrix)

        # default_frame_size = 1.0
        # actual_frame_size = default_frame_size + max(abs(np.array(pose_list)[frm_idx+1][:3] - np.array(pose_list)[frm_idx][:3]))
        # ax.set_xlim((translation[0] - actual_frame_size/2, translation[0] + actual_frame_size/2)); 
        # ax.set_ylim((translation[1] - actual_frame_size/2, translation[1] + actual_frame_size/2)); 
        # ax.set_zlim((translation[2] - actual_frame_size/2, translation[2] + actual_frame_size/2))


        margin = 0.25
        
        # xlim_lower = min(min(start_position_list[frm_idx][:3]), min(target_list[frm_idx]))
        # xlim_upper = max(max(start_position_list[frm_idx][:3]), max(target_list[frm_idx]))


        if animation_camera == ("static"):
            xlim =  ylim = zlim = plotlims


        elif animation_camera == "follow_goal":

            xlim = (translation[0] - margin, translation[0]+ margin)
            ylim = (translation[1] - margin, translation[1]+ margin)
            zlim = (translation[2] - margin, translation[2]+ margin)

        else:
            raise NotImplementedError()

        ax.set_xlim(xlim) 
        ax.set_ylim(ylim) 
        ax.set_zlim(zlim)



        ax.set_xlabel(f'x={pose[0]:.2f}',size=14)
        ax.set_ylabel(f'y={pose[1]:.2f}',size=14)
        ax.set_zlabel(f'z={pose[2]:.2f}',size=14)
        ax.set_title(f"{i*time_between_frames:.2f}s")
        ax.plot(*target_list[frm_idx], color="pink", marker="o", linestyle="")
    print("Generating animation (takes a long time)...", end="",flush=True)
    ani = FuncAnimation(fig, animate, frames=max_time*desired_fps-1, interval= time_between_frames*1000.0, repeat=False)
    plt.close()
    ani.save(filename="test.mp4", writer="ffmpeg", dpi=300)
    print("done.")

def animate_animationdata_from_cache(pars: RobotParameter, seed_train, seed_enjoy):
    with open(f'cache/airframes_animationdata/{hash(pars)}_{seed_train}_{seed_enjoy}_{pars.task_info["task_name"]}_airframeanimationdata.wb', 'rb') as f:
        animationdata: dict = pickle.load(f)
    # print("--pars comparison--")
    # print(pars)
    # print(animationdata['pars'])
    # print("----")
    assert hash(pars) == hash(animationdata['pars'])
    print("seed_train =",animationdata["seed_train"])
    print("seed_enjoy =",animationdata["seed_enjoy"])
    print("task_info =", animationdata["pars"].task_info)
    print(hash(animationdata['pars']))
    positions = np.array(animationdata["poses"].reshape(-1,6).tolist())
    animate_airframe(pars, positions, animationdata["goal_poses"].reshape(-1,6)[:,:3].cpu().numpy())


def plot_enjoy_report(pars: RobotParameter, seed_train, seed_enjoy):
    with open(f'cache/airframes_animationdata/{hash(pars)}_{seed_train}_{seed_enjoy}_{pars.task_info["task_name"]}_airframeanimationdata.wb', 'rb') as f:
        animationdata: dict = pickle.load(f)
    # print("--pars comparison--")
    # print(pars)
    # print(animationdata['pars'])
    # print("----")
    assert hash(pars) == hash(animationdata['pars'])

    print(list(animationdata.keys()))
    

    fig, axs = plt.subplots(3, 3)
    axs[0,0].plot(animationdata["poses"][:,:,:3].reshape(-1,3).tolist(), linewidth=0.5)
    axs[0,0].legend(["x", "y", "z"])
    axs[0,0].set_title("Position")

    axs[1,0].plot(torch.norm(animationdata["poses"][:,:,:3] - animationdata["goal_poses"][:,:,:3], dim=2).reshape(-1).tolist(), linewidth=0.5)
    axs[1,0].set_yscale("log")
    axs[1,0].set_title("Postion error")


    axs[0,1].plot(animationdata["goal_poses"][:,:,:3].reshape(-1,3).tolist(), linewidth=0.5)
    axs[0,1].set_title("Goal Position")

    axs[0,2].plot(animationdata["action_delta"].reshape(-1).tolist(), linewidth=0.5)
    axs[0,2].set_title("Action Delta")

    axs[1,2].plot(animationdata["action"].reshape(-1,animationdata["action"].shape[2]).tolist(), linewidth=0.5)
    axs[1,2].set_title("Actions")

    axs[2,0].plot(animationdata["angvels"].reshape(-1).tolist(), linewidth=0.5)
    axs[2,0].set_title("Angelvel")

    axs[1,1].plot(animationdata["reward"].reshape(-1).tolist(), linewidth=0.5)
    axs[1,1].set_title("Reward")
    
    from itertools import cycle
    markers = cycle(['o', 'v', 's', 'p', '*', 'h', '+', 'x'])

    axs[2,2].plot(animationdata["rewardcomponent_crs"].reshape(-1).tolist(), label= "bonus_survive",  linewidth=0.5, marker=next(markers), markevery=0.33, alpha=0.51)
    axs[2,2].plot(animationdata["rewardcomponent_dist_cost"].reshape(-1).tolist(), label= "dist_cost",  linewidth=0.5, marker=next(markers), markevery=0.33, alpha=0.51)
    axs[2,2].plot(animationdata["rewardcomponent_tiltage_cost"].reshape(-1).tolist(), label= "tiltage_cost",  linewidth=0.5, marker=next(markers), markevery=0.33, alpha=0.51)
    axs[2,2].plot(animationdata["rewardcomponent_velocity_cost"].reshape(-1).tolist(), label= "velocity_cost",  linewidth=0.5, marker=next(markers), markevery=0.33, alpha=0.51)
    axs[2,2].plot(animationdata["rewardcomponent_angularvel_cost"].reshape(-1).tolist(),  label= "angularvel_cost",  linewidth=0.5, marker=next(markers), markevery=0.33, alpha=0.51)
    axs[2,2].plot(animationdata["rewardcomponent_action_cost"].reshape(-1).tolist(), label= "action_cost",  linewidth=0.5, marker=next(markers), markevery=0.33, alpha=0.51)
    axs[2,2].plot(animationdata["rewardcomponent_action_change_cost"].reshape(-1).tolist(), label= "action_change_cost",  linewidth=0.5, marker=next(markers), markevery=0.33, alpha=0.51)
    axs[2,2].plot(animationdata["rewardcomponent_bonus_reward_completed_task"].reshape(-1).tolist(), label= "bonus_reward_completed_task",  linewidth=0.5, marker=next(markers), markevery=0.33, alpha=0.51)
    axs[2,2].set_title("Reward")
    axs[2,2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2, prop={'size': 6})



    axs[2,1].plot(animationdata["completedrewardcomponent_pos_completed_reward"].reshape(-1).tolist(), label= "pos",  linewidth=0.5, marker=next(markers), markevery=0.33, alpha=0.51)
    axs[2,1].plot(animationdata["completedrewardcomponent_linvels_completed_reward"].reshape(-1).tolist(), label= "linvels",  linewidth=0.5, marker=next(markers), markevery=0.33, alpha=0.51)
    axs[2,1].plot(animationdata["completedrewardcomponent_angvels_completed_reward"].reshape(-1).tolist(), label= "angvels",  linewidth=0.5, marker=next(markers), markevery=0.33, alpha=0.51)
    axs[2,1].plot(animationdata["completedrewardcomponent_action_reward"].reshape(-1).tolist(), label= "actions",  linewidth=0.5, marker=next(markers), markevery=0.33, alpha=0.51)
    axs[2,1].set_title("Completed reward")
    axs[2,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2, prop={'size': 6})


    plt.tight_layout()
    plt.savefig("test_control_report.pdf")
    plt.close()



if __name__ == "__main__":


    # # Single rotor interactive plot
    # plot_airframe_interactive_single_rotor()
    # exit(0)


    # # Plot hexarotor simmetric random drone, with 45 degree rotation and translation
    # rs = np.random.RandomState(5)
    # og_pars = rs.random(15)
    # decoded_pars = _decode_symmetric_hexarotor_to_RobotParameter(og_pars)
    # rotation_matrix = np.array([
    #     [1/sqrt(2) , -1/sqrt(2) , 0],
    #     [1/sqrt(2) , 1/sqrt(2) , 0],
    #     [0 , 0 , 1]
    # ])
    # translation = np.array([0.75,0,0])
    # plot_airframe_design(decoded_pars, translation, rotation_matrix)



    # target = [2.3,0.75,1.5]
    # target_str = '[' + ','.join([str(el) for el in target]) + ']'

    # rs = np.random.RandomState(5)
    # og_pars = rs.random(15)
    # pars_str = '[' + ','.join([str(el) for el in og_pars]) + ']'
    # output = subprocess.check_output(f"python src/airframes_objective_functions.py {pars_str} {target_str}", shell=True, text=True)
    # print(output.split("result:")[-1])

 

    # # # Best solution
    x = np.array([0.812831878188605, 0.7541766185236307, 0.44001577293301847, 0.28416197596075315, 0.17072506117811123, 0.40082142384356667, 0.7607302589382322, 0.44599367555068237, 0.758455111661179, 0.5729163603112377, 0.5316279602543038, 0.9463401018817116, 0.48546285370531017, 1.0, 0.22359908694914238])
    pars = _decode_symmetric_hexarotor_to_RobotParameter(x)

    # # # quad
    # pars = quad_pars

    # # hex
    # pars = hex_pars

    pars.task_info = {"task_name":"sphere"}

    # plot_airframe_design(pars)
    seed_train = 15794577
    seed_enjoy = 7460264
    train_and_enjoy = True
    if train_and_enjoy:
        info_dict = motor_rl_objective_function(pars, seed_train, seed_enjoy, 360)
        f = loss_function(info_dict)
        dump_animation_info_dict(pars, seed_train, seed_enjoy, info_dict)
        print("--------------------------")
        print("f(x) = ", f)
        if 'x' in locals():
            [print(f"g_{i}(x) = ", el) for i,el in  enumerate(constraint_check_welf(pars))]
        print("--------------------------")

    plot_enjoy_report(pars, seed_train, seed_enjoy)
    animate_animationdata_from_cache(pars)

