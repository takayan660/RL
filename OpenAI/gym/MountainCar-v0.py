import gym
#from gym import envs
#import numpy as np

env = gym.make('MountainCar-v0')
#env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
for i_episode in range(100):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, _, done, info = env.step(action)
        reward = 1 / (1+(0.5-observation[0])**2)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
