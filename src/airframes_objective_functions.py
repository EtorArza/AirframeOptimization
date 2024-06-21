import numpy as np
import subprocess
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
import pickle
import os
import isaacgym
from aerial_gym_dev.utils.robot_model import RobotParameter, RobotModel
from aerial_gym_dev.utils import analyze_robot_config
import fcl
from pytorch3d.transforms import euler_angles_to_matrix, quaternion_to_matrix, matrix_to_euler_angles
from aerial_gym_dev import AERIAL_GYM_ROOT_DIR
from aerial_gym_dev.utils.urdf_creator import create_urdf_model_for_collision
import torch
import random

def constraint_check_welf(pars: RobotParameter):
    robot = RobotModel(pars)
    check1, check2 = analyze_robot_config.analyze_robot_config(robot)
    return (check1, check2)


def repair_position_device(offset_position_tensor, offset_quat_tensor, og_p, og_euler_degrees):
    new_position = offset_position_tensor + torch.tensor(og_p)
    og_rot = euler_angles_to_matrix(torch.tensor(og_euler_degrees) / 360.0*(2.0*torch.pi), "XYZ")
    repaired_rot = quaternion_to_matrix(offset_quat_tensor)@og_rot
    repaired_angles_degree = matrix_to_euler_angles(repaired_rot, "XYZ") / (2.0*torch.pi) * 360.0
    return new_position.tolist(), repaired_angles_degree.tolist()


def deterministic_simulation(func):
    def wrapper(*args, **kwargs):
        # Save random states
        python_random_state = random.getstate()
        np_random_state = np.random.get_state()
        torch_random_state = torch.get_rng_state()
        if torch.cuda.is_available():
            torch_cuda_random_state = torch.cuda.get_rng_state()
        
        # Set seeds
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Restore random states
        random.setstate(python_random_state)
        np.random.set_state(np_random_state)
        torch.set_rng_state(torch_random_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_random_state)
        
        return result
    
    return wrapper

@deterministic_simulation
def check_collision_and_repair_isaacgym(pars: RobotParameter):

    from isaacgym import gymapi, gymtorch
    import time
    import copy

    repaired_pars = copy.deepcopy(pars)
    robot_model = RobotModel(pars)
    open_visualization = False

    urdf_path = AERIAL_GYM_ROOT_DIR + "/resources/robots/generalized_aerial_robot/generalized_model_constraints.urdf"


    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.dt = 1.0 / 3000.0
    sim_params.substeps = 8
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 8


    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    if open_visualization:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    # plane_params = gymapi.PlaneParams()
    # plane_params.normal = gymapi.Vec3(0, 0, 1)
    # gym.add_ground(sim, plane_params)

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)  # Cast to int
    asset_options.armature = 0.01
    asset_options.linear_damping = 10000.0
    asset_options.angular_damping = 10000.0
    env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, -1.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)



    handle_list = []
    for i in range(0,7):
        create_urdf_model_for_collision(RobotModel(pars), urdf_path, i, 0.045, 0.08)
        if i > 0:
            asset_options.fix_base_link = False
        else:
            asset_options.fix_base_link = True

        component_asset = gym.load_asset(sim, "", urdf_path, asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        component_handle = gym.create_actor(env, component_asset, pose, f"component_{i}", 1, 0)
        handle_list.append(component_handle)

    if open_visualization:
        # Adjust camera position and zoom out
        cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)  # Move the camera further out
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)  # Target the origin
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


    # # body_parts = gym.get_asset_rigid_body_shape_indices(robot_asset)
    # properties = gym.get_asset_rigid_shape_properties(robot_asset)
    # for i in range(len(properties)):
    #     properties[i].filter = 0b0101
    # gym.set_asset_rigid_shape_properties(robot_asset, properties)



    # Add a delay to ensure the viewer has time to render
    total_repair_translation_and_rotation = 0.0

    # print("----------")
    # for handle in handle_list:
    #     print(gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_POS)[0])


    for step in range(100):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_net_contact_force_tensor(sim)
    
        for i in range(len(handle_list[1:])):
            repaired_offset_state = gym.get_actor_rigid_body_states(env, handle_list[1:][i], gymapi.STATE_POS)[0][0]
            offset_position_tensor = torch.tensor([repaired_offset_state[0]['x'], repaired_offset_state[0]['y'],repaired_offset_state[0]['z']])
            offset_quat_tensor = torch.tensor([repaired_offset_state[1]["w"], repaired_offset_state[1]["x"], repaired_offset_state[1]["y"], repaired_offset_state[1]["z"]])


            repaired_pars.motor_translations[i], repaired_pars.motor_orientations[i] = repair_position_device(offset_position_tensor, offset_quat_tensor, pars.motor_translations[i], pars.motor_orientations[i])
            total_repair_translation_and_rotation += torch.sum(torch.abs(offset_position_tensor)) + torch.sum(torch.abs(offset_quat_tensor[:3]))
        contact_forces = gym.acquire_net_contact_force_tensor(sim)
        total_repair_translation_and_rotation += torch.sum(torch.abs(gymtorch.wrap_tensor(contact_forces)))

    if open_visualization:
        for k in range(100000):
            time.sleep(0.03)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)



    if open_visualization:
        gym.destroy_viewer(viewer)
    
    # gym.write_viewer_image_to_file(viewer, filepath)
    # print("Image saved to", filepath)
    gym.destroy_sim(sim)
    return repaired_pars, total_repair_translation_and_rotation.item() > 1e-10



def save_robot_pars_to_file(pars):
    print("save parameters to aerial_gym_dev/envs/base/tmp/config")

    from aerial_gym_dev import AERIAL_GYM_ROOT_DIR
    with open(AERIAL_GYM_ROOT_DIR + "/aerial_gym_dev/envs/base/tmp/config", "wb") as file:
        pickle.dump(pars, file)


def motor_position_enjoy(seed_enjoy):
    cmd_str = f"python src/airframes_objective_functions.py --motor_RL_control_enjoy {seed_enjoy}"
    from datetime import datetime
    import torch
    current_time = datetime.now()
    print(f">> run shell on {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n{cmd_str}")
    output = subprocess.check_output(cmd_str, shell=True, text=True)
    info_dict = torch.load("info_dict.pt")
    return info_dict
    

def _motor_position_enjoy(seed_enjoy):
    from aerial_gym_dev import AERIAL_GYM_ROOT_DIR
    import os
    import isaacgym
    from aerial_gym_dev.utils import  get_args, task_registry, Logger
    import numpy as np
    import torch
    from sample_factory.enjoy_ros import NN_Inference_ROS
    from sf_examples.aerialgym_examples.train_aerialgym import parse_aerialgym_cfg, register_aerialgym_custom_components
    import time
    import torch
    from aerial_gym_dev.envs import base
    from aerial_gym_dev.envs.base.generalized_aerial_robot_config import GenAerialRobotCfg
    from pprint import pprint

    def play(args):

        args.headless = True
        num_airframes_parallel = int(2e4)

        args.num_envs = num_airframes_parallel
        cfg = parse_aerialgym_cfg(evaluation=True)
        cfg.num_agents = num_airframes_parallel
        cfg.eval_deterministic = True
        cfg.train_dir = "./train_dir"
        nn_model = NN_Inference_ROS(cfg)
        print("CFG is:", cfg)



        env_cfg = task_registry.get_cfgs(name=args.task)
        
        env_cfg.control.controller = "no_control"

        # prepare environment
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        rs = np.random.RandomState(seed_enjoy)
        counter = 0
        reset_every = 750

        torch.random.manual_seed(rs.randint(int(1e8)))
        obs, _ = env.reset()
        # random_action = torch.asarray(nn_model.action_space.sample(), device=env.device)
        # obs, _,_,_,_ = env.step(random_action.detach())

        max_steps = 1*int(env.max_episode_length)
        env.record_info_during_reward(2, max_steps, env.action_input.shape[1], torch.device(torch.cuda.current_device()))


        for i in tqdm(range(max_steps)):
            if counter == 0:
                start_time = time.time()
            counter += 1
            action = nn_model.get_action(obs)
            obs, priviliged_obs, rewards, resets, extras = env.step(action)
            if not args.headless:
                env.render()
            if counter % reset_every == 0:
                torch.random.manual_seed(rs.randint(int(1e8)))
                obs, _ = env.reset()

        info_dict = env.infoTensordict

        # reward_list = np.array(reward_list).T
        # pose_list = np.vstack([np.array(pose_list)[:, i, :] for i in range(np.array(pose_list).shape[1])])
        # target_list = np.vstack([np.array(target_list)[:, i, :] for i in range(np.array(target_list).shape[1])])

        return info_dict


    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    args.task = "gen_aerial_robot"
    info_dict = play(args)
    with open("info_dict.pt", "wb") as f:
        torch.save(info_dict,f) 


def motor_position_train(seed_train, train_for_seconds):
    cmd_str = f"python src/airframes_objective_functions.py --motor_RL_control_train --seed={seed_train} --train_for_seconds={train_for_seconds}"
    from datetime import datetime
    current_time = datetime.now()
    print(f">> run shell on {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n{cmd_str}")
    output = subprocess.check_output(cmd_str, shell=True, text=True)

def _motor_position_train(cmdl_args):
    from datetime import datetime
    current_time = datetime.now()
    subprocess.run("rm train_dir/ -rf", shell=True)
    cmd_str = f'python3 ../../sample-factory/sf_examples/gen_aerial_robot_population/train_individual.py --env=gen_aerial_robot {cmdl_args}'
    print(f">> run shell on {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n{cmd_str}", file=sys.stderr)
    output = subprocess.check_output(cmd_str, shell=True, text=True)
    print(output)

def dump_animation_info_dict(pars, seed_train, seed_enjoy, info_dict):
        with open(f'cache/airframes_animationdata/{hash(pars)}_{seed_train}_{seed_enjoy}_{pars.task_info["task_name"]}_airframeanimationdata.wb', 'wb') as f:
            res = {"pars":pars,
                "task_info": pars.task_info,
                "seed_train":seed_train, 
                "seed_enjoy": seed_enjoy,
                **info_dict,
            }
            pickle.dump(res, f)

def log_detailed_evaluation_results(pars, info_dict, seed_train, seed_enjoy, train_for_seconds):
        task_name = pars.task_info["task_name"]
        logpath = f"results/data/details_every_evaluation_{task_name}.csv"
        header = "hash;train_for_seconds;seed_train;seed_enjoy;f;nWaypointsReached;nResets;nWaypointsReached/nResets;total_energy/nWaypointsReached;total_energy\n"
        import torch
        if not os.path.exists(logpath) or os.path.getsize(logpath) == 0:
            with open(logpath, 'w') as file:
                file.write(header)
        with open(logpath, 'a') as file:
            print(f"{hash(pars)};{train_for_seconds};{seed_train};{seed_enjoy};{loss_function(info_dict)};{(info_dict['f_nWaypointsReached']).cpu().item()};{(info_dict['f_nResets']).cpu().item()};{(info_dict['f_nWaypointsReached']/info_dict['f_nResets']).cpu().item()};{(info_dict['f_total_energy']/torch.clamp(info_dict['f_nWaypointsReached'], min=1.0)).cpu().item()};{(info_dict['f_total_energy']).cpu().item()}", file=file)

def motor_rl_objective_function(pars, seed_train, seed_enjoy, train_for_seconds):
    save_robot_pars_to_file(pars)
    motor_position_train(seed_train, train_for_seconds)
    info_dict = motor_position_enjoy(seed_enjoy)
    log_detailed_evaluation_results(pars, info_dict, seed_train, seed_enjoy, train_for_seconds)
    dump_animation_info_dict(pars, seed_train, seed_enjoy, info_dict)
    return info_dict

def loss_function(info_dict):
    return -(
        (info_dict["f_waypoints_reached_energy_adjusted"][0]  / info_dict["f_nResets"][0])
        ).cpu().item()

if __name__ == '__main__':
    # Call objective function from subprocess. Assumes robotConfigFile.txt has been previously written.
    import sys
    if sys.argv[1] == "--motor_RL_control_train":
        assert len(sys.argv) == 4
        assert "seed" in sys.argv[2]
        assert "train_for_seconds" in sys.argv[3]
        sys.argv[1] = "--env=gen_aerial_robot"
        _motor_position_train(sys.argv[2] + " " + sys.argv[3])
        exit(0)

    if sys.argv[1] == "--motor_RL_control_enjoy":
        assert len(sys.argv) == 3
        sys.argv[1] = "--env=gen_aerial_robot"
        seed_enjoy = int(sys.argv[2])
        sys.argv.remove(sys.argv[2])
        _motor_position_enjoy(seed_enjoy)
        exit(0)

    
    else:
        print("LQR has been deprecated.")
        exit(0)