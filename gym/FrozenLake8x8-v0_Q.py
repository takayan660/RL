import gym
from gym import envs
import numpy as np
import sys

nstates = 8*8   # 状態数
nactions = 4    # 行動数
eM = 100000      # 評価を行うエピソード数
alpha = 0.7
gamma = 0.9
goal = 0        # ゴールした回数

class Qlearning:
    """ class for Q Learning """

    @classmethod
    def choose_action(self, Q, new_observation, E_GREEDY_RATIO):
        """ e-greedy法で行動を決める. """
        if E_GREEDY_RATIO < np.random.uniform():
            # greedy法を適用する
            # return self.choose_action_greedy(Q, new_observation)
            return np.argmax(Q[new_observation,:])
        else:
            return np.random.randint(env.action_space.n)

    @classmethod
    def Qupdate(self, Q, new_observation, observation, action, reward):
        """ Q値の更新 """
        return (Q[observation][action] + alpha*(reward-Q[observation][action]+gamma*np.max(Q[new_observation,:])))

if __name__=='__main__':
    Ql = Qlearning
    Q = np.zeros([nstates,nactions])
    E_GREEDY_RATIO = 1.0
    env = gym.make('FrozenLake8x8-v0')
    for i_episode in range(eM):
        observation = env.reset()
        E_GREEDY_RATIO *= 0.9999
        if E_GREEDY_RATIO < 1e-5:
            E_GREEDY_RATIO = 0
        print("Episode {}".format(i_episode+1))
        for t in range(10000):
            env.render()

            # 行動aの獲得
            # action = env.action_space.sample()
            action = Ql.choose_action(Q, observation, E_GREEDY_RATIO)

            # 実行
            new_observation, reward, done, info = env.step(action)

            # Q学習の更新
            Q[observation][action] = Ql.Qupdate(Q, new_observation, observation, action, reward)

            #if np.max(Q) < 1e-4 and t > 100:
            #    print("ERROR: Q=0")
            #    sys.exit()

            # 状態と行動の記録
            observation = new_observation

            if new_observation == nstates-1:
                goal += 1

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    print(Q)
    print(goal)

    env.close()
