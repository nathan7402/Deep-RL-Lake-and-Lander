import gym
from gym import wrappers, logger

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


env = gym.make('LunarLander-v2')

outdir = "../videos/PPO2AgentFrozenLake"
env = wrappers.Monitor(env, directory=outdir, force=True)

env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)

#obs = env.reset()
#for i in range(1000):
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
#    env.render()

logger.set_level(logger.INFO)

for i in range(1000):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if done:
            break
