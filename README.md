# AI Agents to play Q*bert
##### Nathan Williams and Mike Kolor, CS 182, Fall 2018

For our final project, we designed agents to play games from the OpenAI gym.  We chose Frozen Lake and Lunar Lander for their simple structure, in order to make testing feasible on our laptops.

## Setup

Our project is based on the [stable-baselines](https://github.com/hill-a/stable-baselines) fork of the [OpenAI Gym](https://github.com/openai/gym).  Documentation is available [here](https://stable-baselines.readthedocs.io/en/master/).  You MUST be running Python 3.5; we used Python 3.5.2.

To install on Ubuntu, you must first install the prerequisites for `gym` according to the instructions found [here](https://github.com/openai/gym#installation).  For us (on Ubuntu 14.04), this meant running the following:

```
apt-get install libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev \
        libsdl2-2.0.0 libsdl2-dev libglu1-mesa libglu1-mesa-dev libgles2-mesa-dev \
        freeglut3 xvfb libav-tools
```

Now install dependencies for `stable-baselines`:

```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3.5-dev zlib1g-dev python3-tk
```

To install the OpenAI gym and stable-baselines itself, run the following command:

```
pip install requirements.txt
```

## Usage

To run the game from the code directory, use the following commands:

```
python lake.py [agent] [timesteps]
python lander.py [agent] [timesteps]
```

\[agent\] is an optional argument that specifies the agent to use.  Available agents are as follows:
* a2c - synchronous a__ actor-critic (A2C)
* acer - ACER
* acktr - ACKTR
* dqn - DQN
* ppo2 - PPO2
* ppo1 - PPO1
* trpo - TRPO

\[timesteps\] specifies how many frames of training to run.  Our testing was performed with 100,000 and 1,000,000 timesteps.
