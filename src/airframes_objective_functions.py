def target_LQR_control(robot_model, target):


    import numpy as np
    import os
    from datetime import datetime
    import time
    import isaacgym

    import aerial_gym_dev.envs.base.generalized_aerial_robot_config 
    aerial_gym_dev.envs.base.generalized_aerial_robot_config.robot_model = robot_model
    aerial_gym_dev.envs.base.generalized_aerial_robot_config.GenAerialRobotCfg.robot_asset.robot_model = robot_model
    from aerial_gym_dev.utils.urdf_creator import create_urdf_from_model

    from aerial_gym_dev import AERIAL_GYM_ROOT_DIR
    file_path = AERIAL_GYM_ROOT_DIR + "/resources/robots/generalized_aerial_robot/generalized_model_wrench.urdf"
    create_urdf_from_model(robot_model, file_path)

    import aerial_gym_dev.envs.base.generalized_aerial_robot
    import os
    from aerial_gym_dev.utils.task_registry import task_registry

    

    task_registry.register("gen_aerial_robot", aerial_gym_dev.envs.base.generalized_aerial_robot.GenAerialRobot, aerial_gym_dev.envs.base.generalized_aerial_robot_config.GenAerialRobotCfg())

    from aerial_gym_dev.utils import get_args, task_registry
    import torch

    args = get_args()
    args.num_envs = 1
    args.task = 'gen_aerial_robot'

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print("Number of environments", env_cfg.env.num_envs)
    if "lee" in env_cfg.control.controller :
        env_cfg.num_actions = 4
        command_actions = torch.zeros((env_cfg.env.num_envs, env_cfg.control.num_actions))
        command_actions[:, 0] = 0.0
        command_actions[:, 1] = 0.5
        command_actions[:, 2] = 0.0
    elif env_cfg.control.controller == "LQR_control":
        env_cfg.num_actions = 12
        assert type(target)==list
        assert len(target)==3
        command_actions = torch.tensor(target+[0.,0.,0.7,0.,0.,0.,0.,0.,0.], dtype=torch.float32)
        command_actions = command_actions.reshape((1,env_cfg.control.num_actions))
        command_actions = command_actions.repeat(env_cfg.env.num_envs,1)
    elif env_cfg.control.controller == "no_control":
        env_cfg.num_actions = env_cfg.control.num_actions
        command_actions = torch.tensor([ 1./env_cfg.robot_asset.robot_model.max_u for _ in range(env_cfg.control.num_actions)], dtype=torch.float32)
        command_actions *= 9.81*env_cfg.robot_asset.robot_model.total_mass/env_cfg.control.num_actions
        command_actions = command_actions.reshape((1,env_cfg.control.num_actions))
        command_actions = command_actions.repeat(env_cfg.env.num_envs,1)
    
    
    episode_length = 400
    reward_list = []
    obs_list = []
    for i in range(0, episode_length):
        obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)
        r = rewards[0].item()
        pose = np.array(obs['obs'].cpu())[0][0:7]
        reward_list.append(r)
        obs_list.append(pose)

    env.reset()
    return np.array(reward_list), np.array(obs_list)