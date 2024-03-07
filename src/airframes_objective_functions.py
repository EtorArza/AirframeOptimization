

class persistent_data:
    first_call_lqr = True
    env = None
    env_cfg = None

def target_LQR_control(robot_model, target):
    import torch
    import time
    import isaacgym
    import numpy as np

    if persistent_data.first_call_lqr:
        print("Registering environment.")
        from datetime import datetime

        import aerial_gym_dev.envs.base.generalized_aerial_robot_config 
        aerial_gym_dev.envs.base.generalized_aerial_robot_config.robot_model = robot_model
        aerial_gym_dev.envs.base.generalized_aerial_robot_config.GenAerialRobotCfg.robot_asset.robot_model = robot_model
        from aerial_gym_dev.utils.urdf_creator import create_urdf_from_model
        from aerial_gym_dev import AERIAL_GYM_ROOT_DIR
        file_path = AERIAL_GYM_ROOT_DIR + "/resources/robots/generalized_aerial_robot/generalized_model_wrench.urdf"
        create_urdf_from_model(robot_model, file_path)
        import aerial_gym_dev.envs.base.generalized_aerial_robot
        from aerial_gym_dev.utils.task_registry import task_registry

        task_registry.register("gen_aerial_robot", aerial_gym_dev.envs.base.generalized_aerial_robot.GenAerialRobot, aerial_gym_dev.envs.base.generalized_aerial_robot_config.GenAerialRobotCfg())

        from aerial_gym_dev.utils import get_args, task_registry

        args = get_args()
        args.num_envs = 1
        args.task = 'gen_aerial_robot'

        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        persistent_data.env = env
        persistent_data.env_cfg = env_cfg
        assert env_cfg.control.controller == "LQR_control"
        env_cfg.num_actions = 12
        
        env.reset()
        command_actions = torch.tensor(target+[0.,0.,0.7,0.,0.,0.,0.,0.,0.], dtype=torch.float32)
        command_actions = command_actions.reshape((1,env_cfg.control.num_actions))
        command_actions = command_actions.repeat(env_cfg.env.num_envs,1)
        episode_length = 2
        reward_list = []
        obs_list = []
        _, _, _, _, _ = env.step(command_actions)
        persistent_data.first_call_lqr = False
    else:
        print("Reusing environment.")
        env = persistent_data.env
        env_cfg = persistent_data.env_cfg
        

    assert type(target)==list
    assert len(target)==3
    env.reset()
    command_actions = torch.tensor(target+[0.,0.,0.7,0.,0.,0.,0.,0.,0.], dtype=torch.float32)
    command_actions = command_actions.reshape((1,env_cfg.control.num_actions))
    command_actions = command_actions.repeat(env_cfg.env.num_envs,1)
    
    
    episode_length = 300
    reward_list = []
    obs_list = []
    for i in range(0, episode_length):
        obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)
        # env.render()
        r = rewards[0].item()
        pose = np.array(obs['obs'].cpu())[0][0:7]
        reward_list.append(r)
        obs_list.append(pose)

    env.reset()
    return np.array(reward_list), np.array(obs_list)