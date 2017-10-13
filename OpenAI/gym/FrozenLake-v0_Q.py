import gym
from gym import envs
import numpy as np
import sys

state_num = 4*4    # 状態数
action_num = 4      # 行動数
epiode_num = 10000  # 評価を行うエピソード数
step = 1000
policySelect = 1    # 0: e-greedy手法 1: softmax手法
alpha = 0.7
gamma = 0.9
tau = 0.0016        # softmax手法の温度
goal = 0            # 最後の100回中ゴールした回数
goal_total = 0

class Qlearning:
    """ class for Q Learning """
    @classmethod
    def choose_action(self, Q, new_state, epsilon):
        """ 行動決定 """
        if policySelect == 0: # e-greedy
            if epsilon < np.random.uniform():
                # greedy法を適用する
                return np.argmax(Q[new_state,:])
            else:
                return np.random.randint(env.action_space.n)

        elif policySelect == 1: # softmax
            policy = np.exp(Q[new_state,:]/tau) / np.sum(np.exp(Q[new_state,:]/tau))
            
            # 逆関数法で行動選択
            random = np.random.uniform()
            cprob = 0
            for a in range(action_num):
                cprob = cprob + policy[a]
                action = a
                if random < cprob:
                    break

            return action

    @classmethod
    def Qupdate(self, Q, new_state, state, action, reward):
        """ Q値の更新 """
        return (Q[state][action] + alpha*(reward-Q[state][action]+gamma*np.max(Q[new_state,:])))

if __name__=='__main__':
    Ql = Qlearning
    Q = np.zeros([state_num, action_num])
    epsilon = 0.4

    env = gym.make('FrozenLake-v0')

    for i_episode in range(epiode_num):
        state = env.reset()
        epsilon *= 0.999
        if epsilon < 1e-3:
            epsilon = 0.001
        print("Episode {}".format(i_episode+1))
        for t in range(step):
            #env.render()

            # 行動aの獲得
            # action = env.action_space.sample()
            action = Ql.choose_action(Q, state, epsilon)

            # 実行
            new_state, reward, done, _ = env.step(action)

            # Q学習の更新
            Q[state][action] = Ql.Qupdate(Q, new_state, state, action, reward)

            if np.max(Q) < 0 and t > 100:
                print("ERROR: Q=0")
                sys.exit()

            # 状態と行動の記録
            state = new_state

            if done:
                if new_state == state_num-1:
                    goal_total += 1
                    if i_episode > epiode_num-101:
                        goal += 1

                print("Episode finished after {} timesteps".format(t+1))
                break

    print("Q(s,a) =\n{}".format(Q))
    print("Goal total: {}".format(goal_total))
    print("{}/100".format(goal))
    print(epsilon)

    env.close()
