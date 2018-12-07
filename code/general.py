import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DQN, PPO2, A2C

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return True

# Create log dir
log_dir = "../videos/general"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('Qbert-ram-v0')
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

model = A2C(MlpPolicy, env, verbose=1)
# Train the agent
model.learn(total_timesteps=10000, callback=callback)

# video_length = 10000

# env = VecVideoRecorder(env, log_dir,
#                        record_video_trigger=lambda x: x == 0, video_length=video_length,
#                        name_prefix="dqn-qbert-ram-v0")

# for i in range(1000):
#     obs = env.reset()
#     while True:
#         action, _states = model.predict(obs)
#         obs, reward, done, info = env.step(action)
#         if done:
#             break

env.close()
