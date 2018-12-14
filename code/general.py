'''
                AI Agents for OpenAI gym: CS 182 Final Project
--------------------------------------------------------------------------------
                     Nathan Williams and Mike Kolor

To train a specific agent on a game, run the following in the command shell:

python lander.py/lake.py [agent] [timesteps]

[agent] specifies which agent to use.  The following codes are valid:
    ppo2 - ppo2

If an an invalid code or no code is provided, the game defaults to DDPG

[timesteps] specifies how many frames of training to run.  Provided values must
be integers, and the default number is 10,000

'''

import argparse
import sys
import time

from ppo2 import ppo2
from ppo1 import ppo1
from a2c import a2c

def run(env_id, game):
    print("Selected game: {}.".format(env_id))

    # specify timesteps for training
    if len(sys.argv)<3:
        print("No timesteps specified; defaulting to 10,000.")
        timesteps = 10000
    else:
        try:
            timesteps = int(sys.argv[2])
            print("Training with {} timesteps".format(timesteps))
        except ValueError:
            print("Invalid timesteps; must be integer. Defaulting to 10,000.")
            timesteps = 10000

    start_time = time.time()
    # choose agent based on user input and specify directory for output
    if len(sys.argv)<2:
        print("No agent specified; defaulting to PPO2.")
        ppo2(env_id, "../data/ppo2_{}".format(game), timesteps)
    else:
        code = sys.argv[1]
        if code == "ppo2":
            print("PPO2 selected.")
            ppo2(env_id, "../data/ppo2_{}".format(game), timesteps)
        elif code == "ppo1":
            print("PPO1 selected.")
            ppo1(env_id, "../data/ppo1_{}".format(game), timesteps)
        elif code == "a2c":
            print("A2C selected.")
            a2c(env_id, "../data/a2c_{}".format(game), timesteps)
        else:
            print("Invalid code; defaulting to PPO2")
            ppo2(env_id, "../data/ppo2_{}".format(game), timesteps)

    elapsed_time = time.time() - start_time
    print("Training complete!")
    print("Time elapsed: {}.".format(elapsed_time))
    print("See output directory for data.")
