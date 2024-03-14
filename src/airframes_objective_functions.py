import problem_airframes
import numpy as np


def target_LQR_control(target):

    aerial_gym_dev_path="/home/paran/Dropbox/aerial_gym_dev/aerial_gym_dev"
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

    assert type(target)==list
    assert len(target)==3

    task_registry.register("gen_aerial_robot", aerial_gym_dev.envs.base.generalized_aerial_robot.GenAerialRobot, aerial_gym_dev.envs.base.generalized_aerial_robot_config.GenAerialRobotCfg())

    args = get_args()
    args.num_envs = 1
    args.task = 'gen_aerial_robot'
    args.headless = False

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    assert env_cfg.control.controller == "LQR_control"
    env_cfg.num_actions = 12
    
    command_actions = torch.tensor(target+[0.,0.,0.7,0.,0.,0.,0.,0.,0.], dtype=torch.float32)
    command_actions = command_actions.reshape((1,env_cfg.control.num_actions))
    command_actions = command_actions.repeat(env_cfg.env.num_envs,1)
    
    episode_length = 300
    reward_list = []
    obs_list = []
    for i in range(0, episode_length):
        obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)

        if bool(resets.cpu()[0]): # stop if the airframe is reinitialized
            break

        env.render()
        r = rewards[0].item()
        pose = np.array(obs['obs'].cpu())[0][0:7]
        reward_list.append(r)
        obs_list.append(pose)

    env.reset()
    return np.array(reward_list), np.array(obs_list)


if __name__ == '__main__':
    import sys
    assert len(sys.argv) == 2
    target = eval(sys.argv[1])
    res = target_LQR_control(target)
    print("result:")
    print(res[0].tolist())
    print(res[1].tolist())