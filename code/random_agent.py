import os

import gym

from stable_baselines.bench import Monitor

def random_agent(env_id, log_dir, timesteps):
    # Create log dir
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make(env_id)
    env = Monitor(env, log_dir, allow_early_resets=True)

    print("Running episodes with random policy.")

    inc = 0
    done = False
    while inc < timesteps:
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
            inc += 1
            if done:
                break

    env.close()
