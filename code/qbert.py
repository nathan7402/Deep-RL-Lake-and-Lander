import argparse
import sys

import gym
from gym import wrappers, logger

from uselessagent import UselessAgent

'''

code from https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
adapted for Q*bert w/ RAM as input

if len(sys.argv)<2 else sys.argv[1]
'''

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    # choose random action from action_space
    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('Qbert-ram-v0')

    # write video output to chosen directory
    outdir = "../../videos/RandomAgent"
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    if len(sys.argv)<2:
        input_Agent = RandomAgent
    else:
        code = sys.argv[1]
        if code == "useless":
            print("UselessAgent selected.")
            input_agent = UselessAgent
        else:
            print("Invalid code: specify as useless, ...")
            input_agent = RandomAgent
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
