import argparse
import sys

import gym
from gym import wrappers, logger


# an agent that always does nothing


class UselessAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    # choose random action from action_space
    def act(self, observation, reward, done):
        return 0
