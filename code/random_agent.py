import os

import gym

from stable_baselines.bench import Monitor

# an agent that randomly samples an action at each timestep
def random_agent(env_id, log_dir, timesteps):
    # Create log dir
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make(env_id)
    env = Monitor(env, log_dir, allow_early_resets=True)

    print("Running episodes with random policy.")

    # initalize timestep counter
    inc = 0

    while inc < timesteps:
        obs = env.reset()
        while True:
            # choose a random action from action_space
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
            inc += 1
            if done:
                break

    env.close()
