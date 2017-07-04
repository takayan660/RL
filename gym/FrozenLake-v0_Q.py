import gym
from gym import envs
import numpy as np
import sys

nstates = 4*4       # 状態数
nactions = 4        # 行動数
eM = 10000          # 評価を行うエピソード数
alpha = 0.7
gamma = 0.9
policySelect = 2    # 1: e-greedy手法 2: softmax手法
tau = 0.0016        # softmax手法の温度
goal = 0            # ゴールした回数

class Qlearning:
    """ class for Q Learning """

    @classmethod
    def choose_action(self, Q, new_observation, E_GREEDY_RATIO):
        """ 行動決定 """
        if policySelect == 1: # e-greedy
            if E_GREEDY_RATIO < np.random.uniform():
                # greedy法を適用する
                return np.argmax(Q[new_observation,:])
            else:
                return np.random.randint(env.action_space.n)
        elif policySelect == 2: # softmax
            policy = np.exp(Q[new_observation,:]/tau) / np.sum(np.exp(Q[new_observation,:]/tau))
            
            # 逆関数法で行動選択
            random = np.random.uniform()
            cprob = 0
            for a in range(nactions):
                cprob = cprob + policy[a]
                action = a
                if random < cprob:
                    break

            return action

    @classmethod
    def Qupdate(self, Q, new_observation, observation, action, reward):
        """ Q値の更新 """
        return (Q[observation][action] + alpha*(reward-Q[observation][action]+gamma*np.max(Q[new_observation,:])))

if __name__=='__main__':
    Ql = Qlearning
    Q = np.zeros([nstates,nactions])
    E_GREEDY_RATIO = 0.3
    env = gym.make('FrozenLake-v0')
    for i_episode in range(eM):
        observation = env.reset()
        E_GREEDY_RATIO *= 0.999
        if E_GREEDY_RATIO < 1e-4:
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

            if np.max(Q) == 0 and t > 100:
                print("ERROR: Q=0")
                sys.exit()

            # 状態と行動の記録
            observation = new_observation

            if new_observation == nstates-1:
                goal += 1

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    print(Q)
    print(goal)
    print(E_GREEDY_RATIO)

    env.close()
