import argparse
import sys

import gym
from gym import wrappers, logger

from uselessagent import UselessAgent
from randomagent import RandomAgent

'''
                AI Agents for Q*Bert: CS 182 Final Project
--------------------------------------------------------------------------------
                     Nathan Williams and Mike Kolor


To play the game with a specific agent, run the following in the command shell:

python qbert.py [agent]

[agent] specifies which agent to use.  The following codes are valid:
    r - RandomAgent, chooses random action at each step
    u - UselessAgent, always does nothing (action 0)

If an invalid code or no code is provided, the game defaults to RandomAgent.

This code based on https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
adapted for qbert-ram-v0 and any chosen agent.

'''

if __name__ == '__main__':

    # choose agent based on user input and specify directory for output
    if len(sys.argv)<2:
        print("No agent specified; defaulting to RandomAgent.")
        input_agent = RandomAgent
        outdir = "../videos/RandomAgent"
    else:
        code = sys.argv[1]
        if code == "r":
            print("RandomAgent selected.")
            input_agent = RandomAgent
            outdir = "../videos/RandomAgent"
        elif code == "u":
            print("UselessAgent selected.")
            input_agent = UselessAgent
            outdir = "../videos/UselessAgent"
        else:
            print("Invalid code; defaulting to RandomAgent")
            input_agent = RandomAgent
            outdir = "../../videos/RandomAgent"
            outdir = "../videos/RandomAgent"

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('Qbert-ram-v0')

    # write video output to chosen directory

    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    agent = input_agent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
