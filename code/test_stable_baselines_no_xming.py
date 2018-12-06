import gym
from gym import wrappers, logger

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('Qbert-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000)

logger.set_level(logger.INFO)

outdir = "../videos/PPO2Agent"

env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)

for i in range(episode_count):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if done:
            break
