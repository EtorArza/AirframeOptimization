from aerial_gym_dev.envs.base import *
from aerial_gym_dev.utils.task_registry import task_registry, get_args
from aerial_gym_dev import AERIAL_GYM_ROOT_DIR
from aerial_gym_dev.utils.example_configs import PredefinedConfigParameter

import numpy as np

import gymnasium as gym

import pickle

from typing import Dict, List, Optional, SupportsFloat, Tuple, Any, Union

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec, BinaryDiscreteTensorSpec
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, IndependentNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles

import matplotlib.pyplot as plt

import time

def calc_learning_rate_kl(current_lr, recent_kls, num_kls_to_use, lr_schedule_kl_threshold, min_lr, max_lr):
        kls = recent_kls[-num_kls_to_use:]
        mean_kl = np.mean(kls)
        lr = current_lr
        if mean_kl > 2.0 * lr_schedule_kl_threshold:
            lr = max(current_lr / 1.5, min_lr)
        if mean_kl < (0.5 * lr_schedule_kl_threshold):
            lr = min(current_lr * 1.5, max_lr)
        return lr

class AerialGymVecEnv(EnvBase):
    
    batch_locked = False
    
    def __init__(self, aerialgym_env, seed=None):
        self.aerial_env = aerialgym_env
        self.device = self.aerial_env.device
        self._terminated: torch.Tensor = torch.zeros(self.aerial_env.num_envs, dtype=torch.bool)
        self.batch_size = torch.Size([self.aerial_env.num_envs])
        td_params = self.gen_params(self.batch_size)
        self.reward = []
        self.counter = 0
        
        super().__init__(device=self.device, batch_size=self.batch_size)
        self._make_spec(td_params)
        
        if seed is None:
            seed = 8#torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _reset(self, tensordict) -> Dict[str, torch.Tensor]:
        
        if tensordict is None or tensordict.is_empty():
            batch_size = self.batch_size
            obs, priviliged_obs = self.aerial_env.reset()
        
        else:
            batch_size = tensordict.shape
            if any(tensordict["_reset"]):
                obs, priviliged_obs = self.aerial_env.return_obs_after_reset()
            else:
                obs, priviliged_obs = self.aerial_env.reset()
            
        obs = obs["obs"]
        #rot_matrix_robot_state = quaternion_to_matrix(obs[:,[6,3,4,5]])
        #euler_angles = matrix_to_euler_angles(rot_matrix_robot_state, "XYZ")
        #robot_state_euler = torch.cat((obs[:,:3],euler_angles,obs[:,7:]),1)
        #obs = robot_state_euler
        obs_tDict = TensorDict({"obs": obs}, batch_size=batch_size)
        return obs_tDict
    
    def _step(self, tensordict) -> Dict[str, torch.Tensor]:
        actions = tensordict["action"]
        # done vs terminated vs truncated: 
        # terminated means that the episode has finnished and the mdp has reached an end
        # truncated means the the trajectory was stopped early
        # done is a general end of trajectory signal
        
        obs, _, rew, terminated, infos = self.aerial_env.step(actions)
        
        obs = obs["obs"]
        #rot_matrix_robot_state = quaternion_to_matrix(obs[:,[6,3,4,5]])
        #euler_angles = matrix_to_euler_angles(rot_matrix_robot_state, "XYZ")
        #robot_state_euler = torch.cat((obs[:,:3],euler_angles,obs[:,7:]),1)
        #obs = robot_state_euler
        if infos and "time_outs" in infos:
            truncated = infos["time_outs"]
        else:
            truncated = self._terminated
        
        self.reward.append(rew)
        self.counter += 1
        if self.counter % 16 == 0:
            #print("reward: ", torch.mean(torch.tensor(rew)))
            self.reward = []
        
        done = truncated | terminated
        
        out_tDict = TensorDict({"obs": obs,
                                "reward": rew,
                                "done": done.type(torch.bool),
                                "terminated": terminated.type(torch.bool)}, 
                                batch_size=tensordict.shape)
        
        #print("step truncated: ", truncated)
        
        return out_tDict
        
    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
        
    def gen_params(self, batch_size=None) -> TensorDictBase:
        
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "obs_max": self.aerial_env.obs_space.high[:],
                        "obs_min": self.aerial_env.obs_space.low[:],
                        "act_max": self.aerial_env.act_space.high,
                        "act_min": self.aerial_env.act_space.low,
                    },
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        
        return td
        
    def _make_spec(self, td_params):    
        
        self.observation_spec = CompositeSpec(obs =BoundedTensorSpec(low=td_params["params","obs_min"], 
                                                                    high=td_params["params","obs_max"], 
                                                                    shape=torch.Size((*td_params.shape, self.aerial_env.obs_space.shape[0],)),
                                                                    dtype=convert_dtype_np_to_torch(self.aerial_env.obs_space.dtype)), shape=torch.Size((*td_params.shape,)))

        self.state_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(low=td_params["params","act_min"],
                                             high=td_params["params","act_max"],
                                             shape=torch.Size((*td_params.shape, self.aerial_env.act_space.shape[0],)),
                                             dtype=convert_dtype_np_to_torch(self.aerial_env.act_space.dtype))
        
        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size((*td_params.shape,1)))
        
        self.done_spec = BinaryDiscreteTensorSpec(n = 1, shape = torch.Size((*td_params.shape,1)), dtype=torch.bool)

def convert_dtype_np_to_torch(dtype):
    if dtype == np.float32:
        return torch.float32
    elif dtype == np.float64:
        return torch.float64
    elif dtype == np.int32:
        return torch.int32
    elif dtype == np.int64:
        return torch.int64
    else:
        raise ValueError(f"Check dtype conversion between torch and np. Got: {dtype}")
        
def make_aerial_gym_env(seed):
    args = get_args()
    env_name = args.task
    env, env_cfg = task_registry.make_env(env_name, args)
    env_torchrl = AerialGymVecEnv(env,seed=seed)
    return env_torchrl

def set_morphology(morphologyParameter):
        file = open(AERIAL_GYM_ROOT_DIR + "/aerial_gym_dev/envs/base/tmp/config", "wb")
        pickle.dump(morphologyParameter, file)
        file.close()
        
def init_layer(layer, gain=1.0):
    if type(layer) == nn.Linear:
        nn.init.orthogonal_(layer.weight.data, gain=gain)

pars = PredefinedConfigParameter("hex")
set_morphology(pars)

seed = 10 # working: 10->5r ; not working: 4,11,12-> crash 
torch.manual_seed(seed)  # make sure that all torch code is also reproductible
base_env = make_aerial_gym_env(seed)
check_env_specs(base_env)
device = base_env.device

lr = 3e-4
max_grad_norm = 1.0

cyclic_lr = False
obs_norm = True
initialize_policy = False
rollout_length = 5 #5 hex
frames_per_batch = base_env.aerial_env.num_envs*rollout_length
total_frames = 8388608*2
num_batches_per_epoch = 2
sub_batch_size = frames_per_batch // num_batches_per_epoch # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 5 # optimization steps per batch of data collected
clip_epsilon = (0.2) # clip value for PPO loss: see the equation in the intro for more context.)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-2 #1e-4 hex

if obs_norm:
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            VecNorm(in_keys=["obs"]),#,"reward"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
else:
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["obs"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=10000, reduce_dim=[0,1], cat_dim=1)
    
check_env_specs(env)

actor_net = nn.Sequential(
    nn.Linear(env.aerial_env.cfg.env.num_observations, 256, device=device),
    nn.ELU(),
    nn.Linear(256, 128, device=device),
    nn.ELU(),
    nn.Linear(128, 64, device=device),
    nn.ELU(),
    nn.Linear(64, 2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

value_net = nn.Sequential(
    nn.Linear(env.aerial_env.cfg.env.num_observations, 256, device=device),
    nn.ELU(),
    nn.Linear(256, 128, device=device),
    nn.ELU(),
    nn.Linear(128, 128, device=device),
    nn.ELU(),
    nn.Linear(128, 1, device=device),
)

if initialize_policy == True:
    actor_net.apply(init_layer)
    value_net.apply(init_layer)

policy_module = TensorDictModule(
    actor_net, in_keys=["obs"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.action_spec.space.low,
        "max": env.action_spec.space.high,
        "tanh_loc": True,
        "upscale": 1.0,
    },
    return_log_prob=True,
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["obs"],
)

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch*8), #8 hex
    sampler=SamplerWithoutReplacement(shuffle = True), # shuffle = true is default
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=2.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
if cyclic_lr:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, (total_frames // frames_per_batch)//10,1, 0.0)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_frames // frames_per_batch, 0.0)

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

t_collect_start = time.time()
for i, tensordict_data in enumerate(collector):
    t_collect = time.time()-t_collect_start

    t_train_start = time.time()
    
    for _ in range(num_epochs):

        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        
        #print("max loc: ", torch.max(data_view["loc"]))
        #print("min loc: ", torch.min(data_view["loc"]))
        #print("max obs: ", torch.max(data_view["obs"]))
        #print("min obs: ", torch.min(data_view["obs"]))

        if torch.any(data_view["action"].isnan()) | torch.any(data_view["loc"].isnan()) | torch.any(data_view["scale"].isnan()) | torch.any(data_view["obs"].isnan()):
            print("action: ", data_view["action"])
            print("loc: ", data_view["loc"])
            print("scale: ", data_view["scale"])
            print("obs: ", data_view["obs"])
        
        fig, axs = plt.subplots(2,2)
        axs[0,0].hist(data_view["action"].cpu().numpy(), 50, histtype='step', stacked=False, fill=False)
        axs[0,0].set_title("actions")
        axs[0,1].hist(data_view["obs"].cpu().numpy(), 50, histtype='step', stacked=False, fill=False)
        axs[0,1].set_title("observations")
        axs[1,0].hist(data_view["loc"].cpu().numpy(), 50, histtype='step', stacked=False, fill=False)
        axs[1,0].set_title("loc")
        axs[1,1].hist(data_view["scale"].cpu().numpy(), 50, histtype='step', stacked=False, fill=False)
        axs[1,1].set_title("scale")
        plt.show()
        
        t_replay = time.time()
        
        replay_buffer.extend(data_view.cpu()) # this is extremely time consuming
        
        t_mini_batch = time.time()
        for _ in range(frames_per_batch // sub_batch_size):
            t_sample_loss = time.time()
            subdata = replay_buffer.sample(sub_batch_size)
            
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )
            
            #print("loss_value: ", loss_value)
            #print("policy loss: ", loss_vals["loss_objective"])
            #print("exploration loss: ", loss_vals["loss_entropy"])
            #print("value loss: ", loss_vals["loss_critic"])
            
            # Optimization: backward, grad clipping and optimization step
            t_backwards = time.time()
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            t_opt = time.time()
            optim.step()
            optim.zero_grad()

    t_train = time.time()-t_train_start
    logs["value_loss"].append(loss_vals["loss_critic"].mean().item())
    logs["action"].append(tensordict_data["action"].mean(dim=(0,1)).tolist())
    logs["action_mean"].append(tensordict_data["loc"].mean(dim=(0,1)).tolist())
    logs["action_std"].append(tensordict_data["scale"].mean(dim=(0,1)).tolist())
    logs["entropy"].append(loss_vals["entropy"].mean().item())
    logs["obs"].append(torch.max(tensordict_data["obs"]).cpu())
    logs["action_max"].append(torch.max(tensordict_data["loc"]).cpu())
    

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} , "#f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        #f"training time: {t_train: 2.2f} , data collection time: {t_collect: 2.2f}"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 100000000 == 0:
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                #f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                #f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()
    print("learning rate: ", scheduler.get_last_lr())
    
    t_frames = time.time() - t_collect_start
    frames_str = (f"frames/s: {frames_per_batch/t_frames: 2.2f}")
    pbar.set_description(", ".join([eval_str, cum_reward_str, lr_str, frames_str, stepcount_str]))#, stepcount_str, lr_str]))
    
    t_collect_start = time.time()
    
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["value_loss"])
plt.title("value loss")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["entropy"])
plt.title("entropy")
plt.subplot(2, 2, 2)
plt.plot(logs["action_mean"])
plt.title("action_mean")
plt.subplot(2, 2, 3)
plt.plot(logs["action_std"])
plt.title("action_std")
plt.subplot(2, 2, 4)
plt.plot(logs["action"])
plt.title("action")

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["obs"])
plt.title("observations")
plt.subplot(2, 2, 2)
plt.plot(logs["action_max"])
plt.title("action_mean_max")

plt.show()
