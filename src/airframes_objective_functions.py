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
import json
import re

def update_launch_json(command):
    # Path to the launch.json file
    launch_json_path = os.path.join('.vscode', 'launch.json')

    # Read the current content of launch.json
    with open(launch_json_path, 'r') as file:
        data = json.load(file)

    # Find the 'pickle' configuration
    for config in data['configurations']:
        if config['name'] == 'pickle':
            # Extract the script name and arguments from the command
            match = re.search(r'python (.*?) --(.*?) (.*?) (.*?)$', command)
            if match:
                script_path, script_name, args_file, return_file = match.groups()
                
                # Update the 'args' field
                config['args'] = [f"--{script_name}", args_file, return_file]
                
                # Update the 'program' field with the absolute path
                config['program'] = os.path.abspath(script_path)
            
            break

    # Write the updated content back to launch.json
    with open(launch_json_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Updated {launch_json_path}")


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
            
            update_launch_json(command)

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
    for i in range(0,pars.n_motors+1):
        create_urdf_model_for_collision(RobotModel(pars), urdf_path, i, 0.02, 0.12)
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
        torch.clip(mu, min = -1.0, max = 1.0)
        return mu

def model_to_onnx():
    dirs = glob.glob(f"{AERIAL_GYM_ROOT_DIR}/aerial_gym_dev/rl_training/rl_games/runs/gen_ppo_*")
    assert len(dirs) == 1, "There should be exactly one directory that contains the policy"
    path_speed = os.path.join(dirs[0], 'nn', 'best_speed.pth')
    path_efficiency = os.path.join(dirs[0], 'nn', 'best_efficiency.pth')
    path_hover = os.path.join(dirs[0], 'nn', 'best_hover.pth')



    if not os.path.exists(path_hover) and not(os.path.exists(path_speed) and os.path.exists(path_efficiency)):
        raise FileNotFoundError("The file 'best_hover.pth' or ('best_speed.pth' and 'best_efficiency.pth) don't not exist. Cannot proceed with model conversion.")
    for f in [path_speed, path_efficiency, path_hover]:
        if os.path.exists(f):
            print("converting model", f,"to .onnx")
            _model_to_onnx(f)

@run_in_subprocess()
def _model_to_onnx(model_path):

    import yaml
    from rl_games.torch_runner import Runner
    import rl_games.algos_torch.flatten as flatten
    from aerial_gym_dev.rl_training.rl_games.runner import update_config, get_args
    
    args = vars(get_args())
    args['params'] = {'algo_name': 'a2c_continuous'}

    args.update({
        'task': 'position_setpoint_task',
        'checkpoint': model_path,
        'train': False,
        'training': False,
        'seed': 2,
        'play': True,
    })

    yaml_config = os.path.join(AERIAL_GYM_ROOT_DIR, "aerial_gym_dev/rl_training/rl_games/ppo_aerial_quad.yaml")
    with open(yaml_config, "r") as stream:
        config = yaml.safe_load(stream)

    args["seed"] = 2
    config = update_config(config, args)
    config["params"]["load_checkpoint"] = True

    runner = Runner()

    try:
        runner.load(config)
    except yaml.YAMLError as exc:
        print(exc)
        exit(1)
    
    agent = runner.create_player()
    agent.restore(model_path)

    wrapped_model = ModelWrapper(agent.model)
    dummy_input = torch.randn(1, *agent.obs_shape).to(agent.device)
    onnx_model_path = model_path.split("/")[-1].split(".pth")[0] + ".onnx"
    
    print(f"saving {onnx_model_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_model_path,
            verbose=True,
            input_names=['obs'],
            output_names=['mu'],
            dynamic_axes={'obs': {0: 'batch_size'}, 'mu': {0: 'batch_size'}},
            opset_version=11
        )

def update_task_config_parameters(seed: int, headless: bool, waypoint_name: str):
    assert isinstance(seed, int), "seed must be an integer"
    assert isinstance(headless, bool), "headless must be a boolean"
    assert isinstance(waypoint_name, str), "waypoint_name must be a string"

    print("updating", seed, headless, waypoint_name)

    file_path_list = [
        f"{AERIAL_GYM_ROOT_DIR}/aerial_gym_dev/config/task_config/position_setpoint_with_attitude_control.py",
        f"{AERIAL_GYM_ROOT_DIR}/aerial_gym_dev/config/task_config/hover_task.py",
        ]
    
    for file_path in file_path_list:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        seed_count = headless_count = waypoint_count = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith("seed ="):
                lines[i] = f"    seed = {seed}\n"
                seed_count += 1
            elif line.strip().startswith("headless ="):
                lines[i] = f"    headless = {headless}\n"
                headless_count += 1
            elif line.strip().startswith("waypoint_name ="):
                lines[i] = f'    waypoint_name = "{waypoint_name}"\n'
                waypoint_count += 1
        assert seed_count == 1, "Expected exactly one 'seed' field in the file"
        assert headless_count == 1, "Expected exactly one 'headless' field in the file"
        assert waypoint_count == 1, "Expected exactly one 'waypoint_name' field in the file"
        
        with open(file_path, 'w') as file:
            file.writelines(lines)


def get_hover_policy(animation_data_path):
    # animation_data = load_animation_data_and_policy(animation_data_path)
    # motor_position_enjoy(928378, "best_speed.onnx", animation_data["waypoint_name"], "position_setpoint_task", "headless")
    # save_robot_pars_to_file(animation_data["pars"])
    motor_position_train(9999, 600, "hover", "hover_task", "visualize")
    # model_to_onnx()
    # motor_position_enjoy(9998, "best_hover.onnx", "hover", "hover_task", "save")


@run_in_subprocess()
def motor_position_enjoy(seed_enjoy, policy_path, waypoint_name, task_name, render):

    assert task_name in ("hover_task", "position_setpoint_task")
    assert render in  ("headless", "visualize", "save")
    headless = render not in ("visualize", "save")
    record_video = render == "save"

    if record_video:
        subprocess.run("rm results/figures/animation_pngs/* -f", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
        subprocess.run("rm ./animation*.mp4 -f", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
        headless = False

    update_task_config_parameters(seed_enjoy, headless, waypoint_name)

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
    from aerial_gym_dev.rl_training.rl_games.runner import get_args


    # Ensure CUDA is available for ONNX Runtime
    assert 'CUDAExecutionProvider' in ort.get_available_providers(), "CUDA is not available for ONNX Runtime"

    num_airframes_parallel = int(2e4 if headless else 1)
    print("Averaging", num_airframes_parallel, "environments.")
    args = vars(get_args())
    rl_task_env = task_registry.make_task(task_name, {
        "headless": headless,
        "num_envs": num_airframes_parallel,
        "is_enjoy": True,
        "training": False,
        "train": False,
        "play": True,
        "seed": seed_enjoy,
    })
    obs = rl_task_env.reset()[0]["observations"].contiguous()

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_model_gpu = ort.InferenceSession(policy_path, sess_options, providers=['CUDAExecutionProvider'])
    io_binding = ort_model_gpu.io_binding()

    output_shape = (rl_task_env.sim_env.num_envs, rl_task_env.sim_env.num_env_actions + rl_task_env.sim_env.num_robot_actions)
    actions_gpu = torch.empty(output_shape, dtype=torch.float32, device='cuda:0').contiguous()
    time_last_frame = time.time()
    for i in tqdm(range(1000 if headless or record_video else int(1e8))):
        
        # GPU Inference
        io_binding.bind_input(
            name='obs',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(obs.shape),
            buffer_ptr=obs.data_ptr()
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
        obs = rl_task_env.step(actions=actions_gpu)[0]["observations"].contiguous()

        if not headless:
            if record_video and i > 1:
                rl_task_env.sim_env.IGE_env.gym.step_graphics(rl_task_env.sim_env.IGE_env.sim)
                rl_task_env.sim_env.IGE_env.gym.draw_viewer(rl_task_env.sim_env.IGE_env.viewer.viewer, rl_task_env.sim_env.IGE_env.sim, False)
                time.sleep(0.2)
                rl_task_env.sim_env.IGE_env.gym.write_viewer_image_to_file(rl_task_env.sim_env.IGE_env.viewer.viewer, f"results/figures/animation_pngs/{i:06d}.png")
            else:            
                while time.time() - time_last_frame < 0.01:
                    time.sleep(1.0 / 10000.0)
        time_last_frame = time.time()

        if (i+1) % 500 == 0:
            print("i", i)
            obs = rl_task_env.reset()[0]["observations"].contiguous()

    if record_video:
        cmd1 = "ffmpeg -framerate 100 -pattern_type glob -i 'results/figures/animation_pngs/*.png' -c:v libx264 -pix_fmt yuv420p -vf scale=1920:1080 animation.mp4"
        cmd2 = "ffmpeg -framerate 25 -pattern_type glob -i 'results/figures/animation_pngs/*.png' -c:v libx264 -pix_fmt yuv420p -vf scale=1920:1080 animation_slowmo_x4.mp4"
        subprocess.run(cmd1, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
        subprocess.run(cmd2, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr, text=True)

    info_dict = rl_task_env.get_info()
    return info_dict

@run_in_subprocess()
def motor_position_train(seed_train, max_epochs, waypoint_name, task_name, render):

    assert task_name in ("hover_task", "position_setpoint_task")
    headless = render in ("headless", "save")

    update_task_config_parameters(seed_train, headless, waypoint_name)
    from datetime import datetime
    current_time = datetime.now()


    subprocess.run(f"rm *.onnx -f", shell=True)

    subprocess.run(f"rm {AERIAL_GYM_ROOT_DIR}/aerial_gym_dev/rl_training/rl_games/runs/* -rf", shell=True)
    cmd_str = f"wd=`pwd` && cd {AERIAL_GYM_ROOT_DIR}/aerial_gym_dev/rl_training/rl_games && python runner.py --seed={seed_train} --max_epochs={max_epochs} --task=\"{task_name}\""
    print(f">> run shell on {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n{cmd_str}", file=sys.stderr)
    result = subprocess.run(cmd_str, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
    exit_code =  result.returncode
    
    SUCCESS_EXIT_CODE = 0
    CRASH_EXIT_CODE = 1
    FAILED_TO_LEAR_HOVER_EXIT_CODE = 3
    
    
    if exit_code == SUCCESS_EXIT_CODE:
        return "success"
    elif exit_code == FAILED_TO_LEAR_HOVER_EXIT_CODE:
        print("Early stopped, system failed to learn hover in a reasonable time. No policy saved.")
        return "fail"
    elif exit_code == CRASH_EXIT_CODE:
        raise ChildProcessError("Training produced an unexpected crash.")
    else:
        raise ValueError(f"Exit code {exit_code} not recognized.")


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

def dump_animation_data_and_policy(pars, seed_train, seed_enjoy, info_dict, policy_path):
    
    # Create a BytesIO object to hold the compressed data
    compressed_policy_io = io.BytesIO()
    
    with tarfile.open(fileobj=compressed_policy_io, mode="w:gz") as tar:
        tar.add(policy_path, arcname=os.path.basename(policy_path))
    
    # Get the compressed data as bytes
    compressed_policy_data = compressed_policy_io.getvalue()
    
    res = {
        "pars": pars,
        "waypoint_name": info_dict["waypoint_name"],
        "seed_train": seed_train,
        "seed_enjoy": seed_enjoy,
        "policy_data": compressed_policy_data,
        **info_dict,
    }
    
    filename = f'cache/airframes_animationdata/{hash(pars)}_{seed_train}_{seed_enjoy}_{info_dict["waypoint_name"]}_{policy_path.split(".onnx")[0]}_airframeanimationdata.wb'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as f:
        pickle.dump(res, f)

def load_animation_data_and_policy(animationdata_and_policy_file_path):
    with open(animationdata_and_policy_file_path, 'rb') as f:
        animationdata: dict = pickle.load(f)
    
    compressed_policy_data = animationdata['policy_data']
    
    onnx_files = glob.glob("*.onnx")
    for file in onnx_files:
        os.remove(file)

    
    # Extract the compressed policy to ./train_dir
    with tarfile.open(fileobj=io.BytesIO(compressed_policy_data), mode="r:gz") as tar:
        tar.extractall(path=".")
    
    return animationdata

def log_detailed_evaluation_results(pars, policy_path, info_dict, seed_train, seed_enjoy, max_epochs, evaluation_time, result_file_path):
    waypoint_name = info_dict["waypoint_name"]
    header = "hash;pars_name;policy_path;max_epochs;evaluation_time;seed_train;seed_enjoy;nWaypointsReached;percentage_of_battery_used_in_total;nResets;n_waypoints_per_reset;n_waypoints_reachable_based_on_battery_use\n"
    import torch
    if hasattr(pars, "pars_name"):
        pars_name = pars.pars_name
    else:
        pars_name = ""

    if not os.path.exists(result_file_path) or os.path.getsize(result_file_path) == 0:
        with open(result_file_path, 'w') as file:
            file.write(header)
    with open(result_file_path, 'a') as file:
        print(f"{hash(pars)};{pars_name};{policy_path};{max_epochs};{evaluation_time};{seed_train};{seed_enjoy};{info_dict['nWaypointsReached']};{info_dict['percentage_of_battery_used_in_total']};{info_dict['nResets']};{info_dict['n_waypoints_per_reset']};{info_dict['n_waypoints_reachable_based_on_battery_use']}", file=file)

def motor_rl_objective_function(pars, seed_train, seed_enjoy, max_epochs, waypoint_name, log_detailed_evaluation_results_path, render):

    assert render in  ("headless", "visualize", "save")


    save_robot_pars_to_file(pars)
    t_start = time.time() 

    # exit_flag = "success" 
    exit_flag = motor_position_train(seed_train, max_epochs, waypoint_name, "position_setpoint_task", render)

    if exit_flag == "fail":
        print("Train failed. Skipping evaluation.")
        return 0.0, 0.0

    elif exit_flag == "success":
        model_to_onnx()
        info_dict1 = motor_position_enjoy(seed_enjoy, "best_speed.onnx", waypoint_name, "position_setpoint_task", "headless")
        info_dict2 = motor_position_enjoy(seed_enjoy, "best_efficiency.onnx", waypoint_name, "position_setpoint_task", "headless")

        log_detailed_evaluation_results(pars, "best_speed.onnx", info_dict1, seed_train, seed_enjoy, max_epochs, time.time() - t_start, log_detailed_evaluation_results_path)
        log_detailed_evaluation_results(pars, "best_efficiency.onnx", info_dict2, seed_train, seed_enjoy, max_epochs, time.time() - t_start, log_detailed_evaluation_results_path)
        dump_animation_data_and_policy(pars, seed_train, seed_enjoy, info_dict1, "best_speed.onnx")
        dump_animation_data_and_policy(pars, seed_train, seed_enjoy, info_dict2, "best_efficiency.onnx")

        print(f"Objective function (train + enjoy) time: {time.time() - t_start}")
        print("--------------------------")
        print("Number of Waypoints Reached:", info_dict1['nWaypointsReached'])
        print("Total battery use:", info_dict2['percentage_of_battery_used_in_total'])
        print("Number of Resets:", info_dict1['nResets'], info_dict2['nResets'])
        print("f1 (Waypoints per Reset) =", info_dict1['n_waypoints_per_reset'])
        print("f2 (Waypoints reachable based on battery use) =", info_dict2['n_waypoints_reachable_based_on_battery_use'])
        print("--------------------------")

        if render != "headless":
            motor_position_enjoy(seed_enjoy, "best_speed.onnx", waypoint_name, "position_setpoint_task", render)
        return info_dict1["n_waypoints_per_reset"], info_dict2["n_waypoints_reachable_based_on_battery_use"]
    else:
        raise ValueError("Exit flag value not recognized.")





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