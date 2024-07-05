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
from aerial_gym_dev.utils.urdf_creator import create_urdf_from_model, create_urdf_model_for_collision
import torch
import random
import time
import functools
import subprocess
import sys
import tempfile
import pickle
import os
import glob
from datetime import datetime
import shutil
import tarfile
import yaml
from aerial_gym_dev.rl_training.rl_games.runner import get_args, update_config

# This decorator will run the function in a subprocess. It takes the input arguments, serialize them and call this same function. The code in __main__ is also required.
def run_in_subprocess():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            script_name = func.__name__
            
            # Check if we are in a subprocess to avoid recursion
            if len(sys.argv) > 1 and sys.argv[1] == f'--{script_name}':
                return func(*args, **kwargs)
            
            # Serialize the arguments
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
                pickle.dump((args, kwargs), temp_file)
            args_filename = temp_file.name
            
            # Create a temporary file for the return value
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
                return_filename = temp_file.name
            
            # Generate the command string
            command = f'python {os.path.abspath(__file__)} --{script_name} {args_filename} {return_filename}'
            
            current_time = datetime.now()
            print(f">> run shell on {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n{command}")
            
            # Call the subprocess
            subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
            
            # Load the return value
            with open(return_filename, 'rb') as temp_file:
                return_value = pickle.load(temp_file)
            
            # Clean up the temporary files
            os.remove(args_filename)
            os.remove(return_filename)
            
            return return_value
        
        return wrapper
    return decorator


# decorator
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




def constraint_check_welf(pars: RobotParameter):
    robot = RobotModel(pars)
    check1, check2 = analyze_robot_config.analyze_robot_config(robot)
    return (check1, check2)

def plot_admisible_set(pars: RobotParameter):
    robot = RobotModel(pars)
    analyze_robot_config.visualize_admissible_set_forces(robot)

def repair_position_device(offset_position_tensor, offset_quat_tensor, og_p, og_euler_degrees):
    new_position = offset_position_tensor + torch.tensor(og_p)
    og_rot = euler_angles_to_matrix(torch.tensor(og_euler_degrees) / 360.0*(2.0*torch.pi), "XYZ")
    repaired_rot = quaternion_to_matrix(offset_quat_tensor)@og_rot
    repaired_angles_degree = matrix_to_euler_angles(repaired_rot, "XYZ") / (2.0*torch.pi) * 360.0
    return new_position.tolist(), repaired_angles_degree.tolist()

@deterministic_simulation
@run_in_subprocess()
def check_collision_and_repair_isaacgym(pars: RobotParameter):

    from isaacgym import gymapi, gymtorch
    import time
    import copy

    repaired_pars = copy.deepcopy(pars)
    robot_model = RobotModel(pars)
    open_visualization = False

    urdf_path = AERIAL_GYM_ROOT_DIR + "/resources/robots/generalized_model_constraints.urdf"


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
    print("save parameters to aerial_gym_dev/envs/base/tmp/generalized_model")
    with open(AERIAL_GYM_ROOT_DIR + "/aerial_gym_dev/envs/base/tmp/generalized_model", "wb") as file:
        pickle.dump(pars, file)
    urdf_path = AERIAL_GYM_ROOT_DIR + "/resources/robots/generalized_model.urdf"
    create_urdf_from_model(RobotModel(pars), urdf_path)




class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, obs):
        normalized_obs = self._model.norm_obs(obs)
        input_dict = {"obs": normalized_obs}
        actor_output = self._model.a2c_network(input_dict)
        mu = actor_output[0]
        return mu

def model_to_onnx():
    if not os.path.exists('gen_ppo.pth'):
        raise FileNotFoundError("The file 'gen_ppo.pth' does not exist. Cannot proceed with model conversion.")

    import yaml
    from rl_games.torch_runner import Runner
    import rl_games.algos_torch.flatten as flatten
    args = vars(get_args())
    args['params'] = {'algo_name': 'a2c_continuous'}
    args.update({
        'task': 'position_setpoint_task',
        'checkpoint': 'gen_ppo.pth',
        'train': False,
        'seed': 2,
        'play': True,
    })

    runner = Runner()
    yaml_config = os.path.join(AERIAL_GYM_ROOT_DIR, "aerial_gym_dev/rl_training/rl_games/ppo_aerial_quad.yaml")
    with open(yaml_config, "r") as stream:
        config = yaml.safe_load(stream)
    
    config = update_config(config, args)
    
    try:
        runner.load(config)
    except yaml.YAMLError as exc:
        print(exc)
    
    agent = runner.create_player()
    agent.restore('gen_ppo.pth')

    wrapped_model = ModelWrapper(agent.model)

    # Create a dummy input with a dynamic batch dimension
    dummy_input = torch.randn(1, *agent.obs_shape).to(agent.device)

    # Export the model to ONNX
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        "policy.onnx",
        verbose=True,
        input_names=['obs'],
        output_names=['mu'],
        dynamic_axes={'obs': {0: 'batch_size'}, 'mu': {0: 'batch_size'}},
        opset_version=11
    )


@run_in_subprocess()
def motor_position_enjoy(seed_enjoy, headless):
    import os
    import numpy as np
    import torch
    import onnxruntime as ort
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    import time
    from tqdm import tqdm
    from aerial_gym_dev.sim.sim_builder import SimBuilder
    import yaml
    from aerial_gym_dev.task.task_registry import task_registry

    # Ensure CUDA is available for ONNX Runtime
    assert 'CUDAExecutionProvider' in ort.get_available_providers(), "CUDA is not available for ONNX Runtime"

    num_airframes_parallel = int(2e4 if headless else 1e2)
    print("Averaging", num_airframes_parallel, "environments.")
    args = vars(get_args())
    rl_task_env = task_registry.make_task("position_setpoint_task", {
        "headless": headless,
        "num_envs": num_airframes_parallel,
        "is_enjoy": True,
    })
    rl_task_env.reset()

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_model_gpu = ort.InferenceSession("policy.onnx", sess_options, providers=['CUDAExecutionProvider'])
    io_binding = ort_model_gpu.io_binding()

    output_shape = (rl_task_env.sim_env.num_envs, rl_task_env.sim_env.num_env_actions + rl_task_env.sim_env.num_robot_actions)
    actions_gpu = torch.empty(output_shape, dtype=torch.float32, device='cuda:0').contiguous()

    for i in tqdm(range(1000)):
        obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions_gpu)
        
        # GPU Inference
        obs_tensor = obs["observations"].cuda().contiguous()
        io_binding.bind_input(
            name='obs',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(obs_tensor.shape),
            buffer_ptr=obs_tensor.data_ptr()
        )
        io_binding.bind_output(
            name='mu',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(actions_gpu.shape),
            buffer_ptr=actions_gpu.data_ptr()
        )
        ort_model_gpu.run_with_iobinding(io_binding)


        if i % 500 == 0:
            print("i", i)
            rl_task_env.reset()

    info_dict = rl_task_env.get_info()
    return info_dict

@run_in_subprocess()
def motor_position_train(seed_train, max_epochs):
    from datetime import datetime
    current_time = datetime.now()
     
    subprocess.run(f"rm {AERIAL_GYM_ROOT_DIR}/aerial_gym_dev/rl_training/rl_games/runs/* -rf", shell=True)
    cmd_str = f'wd=`pwd` && cd {AERIAL_GYM_ROOT_DIR}/aerial_gym_dev/rl_training/rl_games && python runner.py --seed={seed_train} --save_best_after={max_epochs // 2} --max_epochs={max_epochs}'
    print(f">> run shell on {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n{cmd_str}", file=sys.stderr)
    subprocess.run(cmd_str, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
    dirs = glob.glob(f"{AERIAL_GYM_ROOT_DIR}/aerial_gym_dev/rl_training/rl_games/runs/gen_ppo_*")
    if len(dirs) == 1:
        subprocess.run(f"cp {os.path.join(dirs[0], 'nn', 'gen_ppo.pth')} gen_ppo.pth", shell=True)
    else:
        print("Error: There should be exactly one directory that contains the policy.")
        exit(1)


@run_in_subprocess()
def plot_airframe_to_file_isaacgym(pars: RobotParameter, filepath: str):

    from isaacgym import gymapi, gymtorch
    import time

    save_robot_pars_to_file(pars)
    urdf_path = AERIAL_GYM_ROOT_DIR + "/resources/robots/generalized_model.urdf"




    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    camera_params = gymapi.CameraProperties()
    camera_params.width = 1024
    camera_params.height = 1024
    viewer = gym.create_viewer(sim, camera_params)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)  # Cast to int
    asset_options.armature = 0.01
    robot_asset = gym.load_asset(sim, '', urdf_path, asset_options)

    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, -2.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.5)
    robot_handle = gym.create_actor(env, robot_asset, pose, "robot", 0, 1)

    # Adjust camera position and zoom out
    cam_pos = gymapi.Vec3(0.25, 0.25, 1.2)  # Move the camera further out
    cam_target = gymapi.Vec3(0.0, 0.0, 0.5)  # Target the origin
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    axes_length = 1.0
    ax_center = np.array([0,0,0.5])
   # X-axis (Red)
    x_points = np.array([[0, 0, 0], [axes_length, 0, 0]]+ax_center, dtype=np.float32)
    x_colors = np.array([[1, 0, 0], [1, 0, 0]]+ax_center, dtype=np.float32)
    gym.add_lines(viewer, env, 1, x_points, x_colors)

    # Y-axis (Green)
    y_points = np.array([[0, 0, 0], [0, axes_length, 0]]+ax_center, dtype=np.float32)
    y_colors = np.array([[0, 1, 0], [0, 1, 0]]+ax_center, dtype=np.float32)
    gym.add_lines(viewer, env, 1, y_points, y_colors)

    # Z-axis (Blue)
    z_points = np.array([[0, 0, 0], [0, 0, axes_length]]+ax_center, dtype=np.float32)
    z_colors = np.array([[0, 0, 1], [0, 0, 1]]+ax_center, dtype=np.float32)
    gym.add_lines(viewer, env, 1, z_points, z_colors)



    # Add a delay to ensure the viewer has time to render
    for _ in range(10):
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        time.sleep(0.1)

    gym.write_viewer_image_to_file(viewer, filepath)
    # print("Image saved to", filepath)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)




import pickle
import os
import shutil
import io
import tarfile

def dump_animation_data_and_policy(pars, seed_train, seed_enjoy, info_dict):
    # Compress folder ./train_dir and dump it together with the rest
    policy_path = "policy.onnx"
    
    # Create a BytesIO object to hold the compressed data
    compressed_policy_io = io.BytesIO()
    
    with tarfile.open(fileobj=compressed_policy_io, mode="w:gz") as tar:
        tar.add(policy_path, arcname=os.path.basename(policy_path))
    
    # Get the compressed data as bytes
    compressed_policy_data = compressed_policy_io.getvalue()
    
    res = {
        "pars": pars,
        "task_name": info_dict["task_name"],
        "seed_train": seed_train,
        "seed_enjoy": seed_enjoy,
        "policy_data": compressed_policy_data,
        **info_dict,
    }
    
    filename = f'cache/airframes_animationdata/{hash(pars)}_{seed_train}_{seed_enjoy}_{info_dict["task_name"]}_airframeanimationdata.wb'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as f:
        pickle.dump(res, f)

def load_animation_data_and_policy(animationdata_and_policy_file_path):
    with open(animationdata_and_policy_file_path, 'rb') as f:
        animationdata: dict = pickle.load(f)
    
    compressed_policy_data = animationdata['policy_data']
    
    if os.path.exists("policy.onnx"):
        os.remove("policy.onnx")
    
    # Extract the compressed policy to ./train_dir
    with tarfile.open(fileobj=io.BytesIO(compressed_policy_data), mode="r:gz") as tar:
        tar.extractall(path=".")
    
    return animationdata

def log_detailed_evaluation_results(pars, info_dict, seed_train, seed_enjoy, max_epochs):
    task_name = info_dict["task_name"]
    logpath = f"results/data/details_every_evaluation_{task_name}.csv"
    header = "hash;max_epochs;seed_train;seed_enjoy;f;nWaypointsReached;nResets;nWaypointsReached/nResets;total_energy/nWaypointsReached;total_energy\n"
    import torch
    if not os.path.exists(logpath) or os.path.getsize(logpath) == 0:
        with open(logpath, 'w') as file:
            file.write(header)
    with open(logpath, 'a') as file:
        print(f"{hash(pars)};{max_epochs};{seed_train};{seed_enjoy};{loss_function(info_dict)};{(info_dict['f_nWaypointsReached']).cpu().item()};{(info_dict['f_nResets']).cpu().item()};{(info_dict['f_nWaypointsReached']/info_dict['f_nResets']).cpu().item()};{(info_dict['f_total_energy']/torch.clamp(info_dict['f_nWaypointsReached'], min=1.0)).cpu().item()};{(info_dict['f_total_energy']).cpu().item()}", file=file)

def motor_rl_objective_function(pars, seed_train, seed_enjoy, max_epochs):
    save_robot_pars_to_file(pars)
    motor_position_train(seed_train, max_epochs)
    model_to_onnx()
    # subprocess.run(f"rm gen_ppo.pth", shell=True)
    info_dict = motor_position_enjoy(seed_enjoy, False)
    log_detailed_evaluation_results(pars, info_dict, seed_train, seed_enjoy, max_epochs)
    dump_animation_data_and_policy(pars, seed_train, seed_enjoy, info_dict)
    return info_dict

def loss_function(info_dict):
    return -(
        (info_dict["f_waypoints_reached_energy_adjusted"][0]  / info_dict["f_nResets"][0])
        ).cpu().item()




# This is just a hack such that the functions with the decorator @run_in_subprocess are executed in a subprocess automagically.
# Main script logic
if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1].startswith('--'):
        func_name = sys.argv[1][2:]  # Remove the '--' prefix
        args_filename = sys.argv[2]
        return_filename = sys.argv[3]
        
        # Dynamically get the function by name
        func = globals().get(func_name)
        if func is None:
            print(f"Error: Function '{func_name}' not found.")
            sys.exit(1)
        
        # Load arguments
        with open(args_filename, 'rb') as temp_file:
            args, kwargs = pickle.load(temp_file)
        
        # Call the function and get the return value
        return_value = func(*args, **kwargs)
        
        # Save the return value
        with open(return_filename, 'wb') as temp_file:
            pickle.dump(return_value, temp_file)
        
        sys.exit(0)
    else:
        # Normal script execution
        pass