import os

import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from stable_baselines import DDPG

def ddpg(env_id, log_dir, timesteps):
    # Create log dir
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make(env_id)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    # noise objects for ddpg
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    model = DDPG(MlpPolicy, env, verbose=0, param_noise=param_noise, action_noise=action_noise)
    # Train the agent
    print("Beginning training episodes with DDPG.")
    model.learn(total_timesteps=timesteps)

    env.close()
