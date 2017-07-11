import gym
from gym import envs
import numpy as np
import sys

state1p_num = 180        # 状態数
state2v_num = 140
action_num = 3          # 行動数
epiode_num = 1000000    # 評価を行うエピソード数
step = 1000
policySelect = 0        # 0: e-greedy手法 1: softmax手法
alpha = 0.7
gamma = 0.9
tau = 0.0016            # softmax手法の温度
goal = 0                # 最後の100回中ゴールした回数
goal_total = 0

class Qlearning:
    """ class for Q Learning """
    @classmethod
    def choose_action(self, Q, new_state, epsilon):
        """ 行動決定 """
        if policySelect == 0: # e-greedy
            if epsilon < np.random.uniform():
                # greedy法を適用する
                return np.argmax(Q[int(new_state[0]), int(new_state[1]),:])
            else:
                return np.random.randint(env.action_space.n)

        elif policySelect == 1: # softmax
            policy = np.exp(Q[int(new_state[0]), int(new_state[1]),:]/tau) / np.sum(np.exp(Q[int(new_state[0]), int(new_state[1]),:]/tau))
            
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
        return (Q[int(state[0]), int(state[1]), action] + alpha*(reward-Q[int(state[0]), int(state[1]), action]+gamma*np.max(Q[int(new_state[0]), int(new_state[1]),:])))

if __name__=='__main__':
    Ql = Qlearning
    Q = np.zeros([state1p_num, state2v_num, action_num])
    epsilon = 1.0

    env = gym.make('MountainCar-v0')

    for i_episode in range(epiode_num):
        state = env.reset()
        epsilon *= 0.9999
        if epsilon < 1e-3:
            epsilon = 0.001
        print("Episode {}".format(i_episode+1))
        for t in range(step):
            env.render()

            # 行動aの獲得
            # action = env.action_space.sample()
            action = Ql.choose_action(Q, state, epsilon)

            # 実行
            new_state, _, done, _ = env.step(action)

            # reward の獲得
            reward = 1 / (1+(0.5-new_state[0])**2)

            # 離散化
            new_state[0] = round(new_state[0]*100)+120
            new_state[1] = round(new_state[1]*100)+70
            
            # Q学習の更新
            Q[int(state[0]), int(state[1]), action] = Ql.Qupdate(Q, new_state, state, action, reward)

            #if np.max(Q) < 0 and t > 100:
            #    print("ERROR: Q=0")
            #    sys.exit()

            # 状態と行動の記録
            state = new_state

            print new_state[0]

            if done:
                if new_state[0] == state1p_num-10:
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
