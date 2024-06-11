import numpy as np
import subprocess
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
import pickle
import os


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
    print(f">> run shell on {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n{cmd_str}")
    output = subprocess.check_output(cmd_str, shell=True, text=True)
    print(output)

def dump_animation_info_dict(pars, seed_train, seed_enjoy, info_dict):
        with open(f'cache/airframes_animationdata/{hash(pars)}_airframeanimationdata.wb', 'wb') as f:
            res = {"pars":pars,
                "seed_train":seed_train, 
                "seed_enjoy": seed_enjoy,
                **info_dict,
            }
            pickle.dump(res, f)

def log_detailed_evaluation_results(pars, info_dict, seed_train, seed_enjoy, train_for_seconds):
        logpath = "results/data/details_every_evaluation.csv"
        header = "hash;train_for_seconds;seed_train;seed_enjoy;f;nWaypointsReached/nResets;total_energy\n"
        if not os.path.exists(logpath) or os.path.getsize(logpath) == 0:
            with open(logpath, 'w') as file:
                file.write(header)
        with open(logpath, 'a') as file:
            print(f"{hash(pars)};{train_for_seconds};{seed_train};{seed_enjoy};{loss_function(info_dict)};{(info_dict['f_nWaypointsReached']/info_dict['f_nResets']).cpu().item()};{(info_dict['f_total_energy']).cpu().item()}",  file=file)

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