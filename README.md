# AI Agents to play Q*bert
##### Nathan Williams and Mike Kolor, CS 182, Fall 2018


For our final project, we designed agents to play the Atari 2600 game Q*bert, using the OpenAI gym as a starting point.

## Setup

For detailed instructions on installing the OpenAI Gym, see https://github.com/openai/gym#installation

For a more basic install, run the following commands

```
pip install gym
pip install -e '.[atari]'
```


## Usage

To run the game from the code directory, use the following command:

```
python qbert.py [agent]
```

[agent] is an optional argument that specifies the agent to use.  Available agents are as follows:
* r - RandomAgent, chooses actions pseudorandomly
* u - UselessAgent, always does nothing
