import os

import gym
import numpy as np

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor

from stable_baselines import DQN

def dqn(env_id, log_dir, timesteps):
    # Create log dir
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make(env_id)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = DQN(MlpPolicy, env, verbose=0)
    # Train the agent
    print("Beginning training episodes with DQN.")
    model.learn(total_timesteps=timesteps)

    env.close()
