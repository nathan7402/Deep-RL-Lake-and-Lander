import os
import sys

import gym

from stable_baselines.bench import Monitor

def policy(obs):
    if obs in [57,58]:
        return 0
    elif obs in [7,15,23,28,31,39,45,47,53,55]:
        return 1
    elif obs in [0,1,2,3,4,5,6,8,9,10,12,13,14,16,20,22,24,27,30,36,37,43,50,60,61,62]:
        return 2
    elif obs in [11,17,18,21,25,26, 32, 33, 34, 38, 40,44,48,51,56]:
        return 3
    elif obs in [19, 29, 35, 41, 42, 46, 49, 52, 54, 59]:
        print("Error: player dead; invalid observation!")
        return 0
    elif obs == 63:
        print("Error: game won; invalid observation!")
        return 0
    else:
        print("Error: invalid observation.  Aborting run.")
        sys.exit(1)

def hardcode(env_id, log_dir, timesteps):
    # Create log dir
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make(env_id)
    env = Monitor(env, log_dir, allow_early_resets=True)

    print("Running episodes with hardcoded policy.")

    inc = 0
    done = False
    while inc < timesteps:
        obs = env.reset()
        while True:
            action = policy(obs)
            obs, _, done, _ = env.step(action)
            inc += 1
            if done:
                break

    env.close()
