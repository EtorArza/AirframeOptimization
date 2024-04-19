import numpy as np
import subprocess
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
import pickle


def save_robot_pars_to_file(pars):
    print("save parameters to robotConfigFile.txt")
    with open('/home/paran/Dropbox/NTNU/11_constraints_encoding/code/robotConfigFile.txt','w') as f:
        print('pars.cq=',pars.cq, file=f, flush=True)
        print('pars.frame_mass=',pars.frame_mass, file=f)
        print('pars.motor_masses=',pars.motor_masses, file=f)

        print('pars.motor_translations=',pars.motor_translations, file=f)
        print('pars.motor_orientations=',pars.motor_orientations, file=f)
        print('pars.motor_directions=',pars.motor_directions, file=f)



        print('pars.max_u=',pars.max_u, file=f)
        print('pars.min_u=',pars.min_u, file=f)


def _target_LQR_control(target_list, render:bool):

    aerial_gym_dev_path="/home/paran/Dropbox/NTNU/aerial_gym_dev/aerial_gym_dev"
    import sys
    import isaacgym
    sys.path.append(aerial_gym_dev_path)
    from aerial_gym_dev import AERIAL_GYM_ROOT_DIR
    file_path = AERIAL_GYM_ROOT_DIR + "/resources/robots/generalized_aerial_robot/generalized_model_wrench.urdf"
    from aerial_gym_dev.utils.urdf_creator import create_urdf_from_model
    from aerial_gym_dev.utils.task_registry import task_registry
    from aerial_gym_dev.utils import get_args, task_registry
    import torch
    import aerial_gym_dev.envs.base.generalized_aerial_robot_config 
    import aerial_gym_dev.envs.base.generalized_aerial_robot

    assert type(target_list)==list
    assert len(target_list[0])==3

    task_registry.register("gen_aerial_robot", aerial_gym_dev.envs.base.generalized_aerial_robot.GenAerialRobot, aerial_gym_dev.envs.base.generalized_aerial_robot_config.GenAerialRobotCfg())

    args = get_args()
    args.num_envs = 1
    args.task = 'gen_aerial_robot'
    args.headless = not render

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    assert env_cfg.control.controller == "LQR_control"
    env_cfg.num_actions = 12
    

    
    reward_list = []
    pose_list = []
    for i in range(0, len(target_list)):

        target = target_list[i]
        command_actions = torch.tensor(target+[0.,0.,0.,0.,0.,0.,0.,0.,0.], dtype=torch.float32)
        command_actions = command_actions.reshape((1,env_cfg.control.num_actions))
        command_actions = command_actions.repeat(env_cfg.env.num_envs,1)

        print(command_actions)

        obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)

        if bool(resets.cpu()[0]): # stop if the airframe is reinitialized
            break

        if render:
            env.render()

        r = rewards[0].item()
        pose = np.array(obs['obs'].cpu())[0][0:7]
        reward_list.append(r)
        pose_list.append(pose)

    # def loss_function(poses, target_list):
    #     f = 0
    #     extra_loss = 0
    #     for i, pose in enumerate(poses):
    #         target = target_list[i]
    #         distance = np.linalg.norm(np.array(target) - pose[0:3])

    #         if i>0 and i % steps_per_target == 0:
    #             extra_loss += distance

    #         f += distance / len(poses) + extra_loss
        
    #     f -= 100000 * (len(poses) - episode_length) # Bonus reward for episode length. Longer episode length means that evaluation was not prematurely terminated.
    #     return f

    env.reset()
    return target_list, pose_list, np.mean(np.array(reward_list))


def motor_position_enjoy(seed_enjoy):
    cmd_str = f"python src/airframes_objective_functions.py --motor_RL_control_enjoy {seed_enjoy}"
    from datetime import datetime
    current_time = datetime.now()
    print(f">> run shell on {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n{cmd_str}")
    output = subprocess.check_output(cmd_str, shell=True, text=True)
    res = eval(output.split("result:\n")[-1].strip("\n"))
    target_list = [np.array(el) for el in res['target_list']]
    pose_list = [np.array(el) for el in res['pose_list']]
    mean_reward = res['mean_reward']


    import torch
    import pytorch3d.transforms as p3d_transforms

    for i in range(len(pose_list)):
        xyzw_quaternion = torch.asarray(pose_list[i][3:])
        quaternion = xyzw_quaternion[[3,0,1,2]]
        rotation = p3d_transforms.quaternion_to_matrix(quaternion)
        euler_angles = p3d_transforms.matrix_to_euler_angles(rotation, "XYZ")
        pose_list[i] = np.concatenate([pose_list[i][:3], euler_angles.cpu().numpy()])

    return target_list, pose_list, mean_reward

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
        cfg.train_dir = "./train_dir"
        nn_model = NN_Inference_ROS(cfg)
        print("CFG is:", cfg)



        env_cfg = task_registry.get_cfgs(name=args.task)
        
        env_cfg.control.controller = "no_control"

        # prepare environment
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        rs = np.random.RandomState(seed_enjoy)
        counter = 0
        reset_every = 800

        torch.random.manual_seed(rs.randint(int(1e8)))
        obs, _ = env.reset()
        # random_action = torch.asarray(nn_model.action_space.sample(), device=env.device)
        # obs, _,_,_,_ = env.step(random_action.detach())

        max_steps = 1*int(env.max_episode_length) 

        pose_list = []
        target_list = []
        reward_list = []

        n_poses_return = 4


        for i in tqdm(range(max_steps)):
            if counter == 0:
                start_time = time.time()
            counter += 1
            action = nn_model.get_action(obs)
            obs, priviliged_obs, rewards, resets, extras = env.step(action)


            target_list.append(env.goal_states[0:n_poses_return, 0:3].cpu().numpy())
            pose_list.append(obs['obs'][0:n_poses_return, 0:7].cpu().numpy())
            reward_list.append(rewards.cpu().numpy().tolist())

            if counter % reset_every == 0:
                torch.random.manual_seed(rs.randint(int(1e8)))
                obs, _ = env.reset()

        reward_list = np.array(reward_list).T

        pose_list = np.vstack([np.array(pose_list)[:, i, :] for i in range(np.array(pose_list).shape[1])])
        target_list = np.vstack([np.array(target_list)[:, i, :] for i in range(np.array(target_list).shape[1])])

        return target_list, pose_list, float(np.mean(np.mean(reward_list)))


    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    args.task = "gen_aerial_robot"
    target_list, pose_list, mean_reward = play(args)
    mean_reward = -mean_reward # Our direct search framework assumes minimization.

    res = {"target_list":[el.tolist() for el in target_list], "pose_list":[el.tolist() for el in pose_list], "mean_reward":mean_reward}

    return res


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
    cmd_str = f'python3 -m sf_examples.gen_aerial_robot_population.train_individual --env=gen_aerial_robot {cmdl_args}'
    print(f">> run shell on {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n{cmd_str}")
    output = subprocess.check_output(cmd_str, shell=True, text=True)
    print(output)


def motor_rl_objective_function(pars, seed_train, seed_enjoy, train_for_seconds):
    save_robot_pars_to_file(pars)
    motor_position_train(seed_train, train_for_seconds)
    target_list, pose_list, mean_reward = motor_position_enjoy(seed_enjoy)
    with open(f'cache/airframes_animationdata/{hash(pars)}_ariframeanimationdata.wb', 'wb') as f:
        res = {"pars":pars, 
               "target_list":[el.tolist() for el in target_list], 
               "pose_list":[el.tolist() for el in pose_list], 
               "mean_reward":mean_reward, 
               "seed_train":seed_train, 
               "seed_enjoy": seed_enjoy}
        pickle.dump(res, f)
    return target_list, pose_list, mean_reward


def target_lqr_objective_function(pars, target_list):
    save_robot_pars_to_file(pars)

    target_str = str(target_list).replace(" ", "")

    cmd_str = f"python src/airframes_objective_functions.py {target_str}"
    from datetime import datetime
    current_time = datetime.now()
    print(f">> run shell on {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n{cmd_str}")
    output = subprocess.check_output(cmd_str, shell=True, text=True)
    rewards = np.array(eval(output.split("result:")[-1].split("\n")[1]))
    pose_list = np.array(eval(output.split("result:")[-1].split("\n")[2]))

    return target_list, pose_list, np.mean(rewards)



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
        res = _motor_position_enjoy(seed_enjoy)
        print("result:")
        print(res)
        exit(0)

    
    else:
        print("Need to fix code so it works with LQR as well.")
        exit(0)

    exit(0)
    [print(el) for el in sys.argv]
    assert len(sys.argv) == 2
    target_list = eval(sys.argv[1])
    res = _target_LQR_control(target_list, False)
    print("result:")
    print(res[0].tolist())
    print(res[1].tolist())