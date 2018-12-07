from gym import wrappers, logger

from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN

env = make_atari('BreakoutNoFrameskip-v4')

outdir = "../../videos/breakout"
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)

model = DQN(CnnPolicy, env, verbose=1)

print("Beginning training episodes")
model.learn(total_timesteps=25000)

logger.set_level(logger.INFO)

for i in range(1000):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if done:
            break
