import gym
from gym import envs
import numpy as np
import sys

nstates = 8*8   # $B>uBV?t(B
nactions = 4    # $B9TF0?t(B
eM = 100000      # $BI>2A$r9T$&%(%T%=!<%I?t(B
alpha = 0.7
gamma = 0.9
goal = 0        # $B%4!<%k$7$?2s?t(B

class Qlearning:
    """ class for Q Learning """

    @classmethod
    def choose_action(self, Q, new_observation, E_GREEDY_RATIO):
        """ e-greedy$BK!$G9TF0$r7h$a$k(B. """
        if E_GREEDY_RATIO < np.random.uniform():
            # greedy$BK!$rE,MQ$9$k(B
            # return self.choose_action_greedy(Q, new_observation)
            return np.argmax(Q[new_observation,:])
        else:
            return np.random.randint(env.action_space.n)

    @classmethod
    def Qupdate(self, Q, new_observation, observation, action, reward):
        """ Q$BCM$N99?7(B """
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

            # $B9TF0(Ba$B$N3MF@(B
            # action = env.action_space.sample()
            action = Ql.choose_action(Q, observation, E_GREEDY_RATIO)

            # $B<B9T(B
            new_observation, reward, done, info = env.step(action)

            # Q$B3X=,$N99?7(B
            Q[observation][action] = Ql.Qupdate(Q, new_observation, observation, action, reward)

            #if np.max(Q) < 1e-4 and t > 100:
            #    print("ERROR: Q=0")
            #    sys.exit()

            # $B>uBV$H9TF0$N5-O?(B
            observation = new_observation

            if new_observation == nstates-1:
                goal += 1

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    print(Q)
    print(goal)

    env.close()
