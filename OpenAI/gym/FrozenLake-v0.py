import gym
from gym import envs
#import numpy as np

env = gym.make('FrozenLake-v0')
env = gym.wrappers.Monitor(env, 'experiment/FrozenLake', force=True)
for i_episode in range(10000):
    observation = env.reset()
    for t in range(10000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
