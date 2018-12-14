'''
                AI Agents for OpenAI gym: CS 182 Final Project
--------------------------------------------------------------------------------
                     Nathan Williams and Mike Kolor

To train a specific agent on lander, run the following in the command shell:

python lander.py [agent] [timesteps]

[agent] specifies which agent to use.  The following codes are valid:
    ddpg - DDPG

If an an invalid code or no code is provided, the game defaults to DDPG

[timesteps] specifies how many frames of training to run.  Provided values must
be integers, and the default number is 10,000

'''

import argparse
import sys

from ddpg_lander import ddpg_lander
from ppo2_lander import ppo2_lander

if __name__ == '__main__':

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

    # choose agent based on user input and specify directory for output
    if len(sys.argv)<2:
        print("No agent specified; defaulting to DDPG.")
        ddpg_lander("../../data/ddpg_lander", timesteps)
    else:
        code = sys.argv[1]
        if code == "ddpg":
            print("DDPG selected.")
            ddpg_lander("../../data/ddpg_lander", timesteps)
        elif code == "ppo2":
            print("PPO2 selected.")
            ppo2_lander("../../data/ppo2_lander", timesteps)
        else:
            print("Invalid code; defaulting to DDPG")
            ddpg_lander("../../data/ddpg_lander", timesteps)

    print("Training complete.  See output directory for data.")

