import os

import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor

from stable_baselines import PPO2

def ppo2_lander(log_dir, timesteps):
    # Create log dir
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make('LunarLanderContinuous-v2')
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=timesteps)

    env.close()
