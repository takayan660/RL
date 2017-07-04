import gym
from gym import envs
import numpy as np
import sys

nstates = 4*4       # $B>uBV?t(B
nactions = 4        # $B9TF0?t(B
eM = 10000          # $BI>2A$r9T$&%(%T%=!<%I?t(B
alpha = 0.7
gamma = 0.9
policySelect = 2    # 1: e-greedy$B<jK!(B 2: softmax$B<jK!(B
tau = 0.0016        # softmax$B<jK!$N29EY(B
goal = 0            # $B%4!<%k$7$?2s?t(B

class Qlearning:
    """ class for Q Learning """

    @classmethod
    def choose_action(self, Q, new_observation, E_GREEDY_RATIO):
        """ $B9TF07hDj(B """
        if policySelect == 1: # e-greedy
            if E_GREEDY_RATIO < np.random.uniform():
                # greedy$BK!$rE,MQ$9$k(B
                return np.argmax(Q[new_observation,:])
            else:
                return np.random.randint(env.action_space.n)
        elif policySelect == 2: # softmax
            policy = np.exp(Q[new_observation,:]/tau) / np.sum(np.exp(Q[new_observation,:]/tau))
            
            # $B5U4X?tK!$G9TF0A*Br(B
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
        """ Q$BCM$N99?7(B """
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

            # $B9TF0(Ba$B$N3MF@(B
            # action = env.action_space.sample()
            action = Ql.choose_action(Q, observation, E_GREEDY_RATIO)

            # $B<B9T(B
            new_observation, reward, done, info = env.step(action)

            # Q$B3X=,$N99?7(B
            Q[observation][action] = Ql.Qupdate(Q, new_observation, observation, action, reward)

            if np.max(Q) == 0 and t > 100:
                print("ERROR: Q=0")
                sys.exit()

            # $B>uBV$H9TF0$N5-O?(B
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
