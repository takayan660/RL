import gym
from gym import wrappers
import numpy as np
import sys

state_num = 40          # $B>uBV?t(B
action_num = 3          # $B9TF0?t(B
epiode_num = 10000      # $BI>2A$r9T$&%(%T%=!<%I?t(B
step = 1000
policySelect = 1        # 0: e-greedy$B<jK!(B 1: softmax$B<jK!(B
alpha = 1.0
gamma = 1.0
tau = 0.04              # softmax$B<jK!$N29EY(B
goal = 0                # $B:G8e$N(B100$B2sCf%4!<%k$7$?2s?t(B
goal_total = 0

def run_episode(env, Q, render=False):
    state = env.reset()
    eps = 0.02
    for _ in range(200):
        if render:
            env.render()
        # $B9TF0(Ba$B$N3MF@(B
        # action = env.action_space.sample()
        action = Ql.choose_action(Q, state, eps)

        # $B<B9T(B
        observation, reward, done, _ = env.step(action)


class Qlearning:
    """ class for Q Learning """
    @classmethod
    def choose_action(self, Q, new_state, eps):
        """ $B9TF07hDj(B """
        if policySelect == 0: # e-greedy
            if eps < np.random.uniform():
                # greedy$BK!$rE,MQ$9$k(B
                return np.argmax(Q[int(new_state[0]), int(new_state[1]),:])
            else:
                return np.random.randint(env.action_space.n)

        elif policySelect == 1: # softmax
            policy = np.exp(Q[int(new_state[0]), int(new_state[1]),:]/tau) / np.sum(np.exp(Q[int(new_state[0]), int(new_state[1]),:]/tau))
            
            # $B5U4X?tK!$G9TF0A*Br(B
            #random = np.random.uniform()
            random = 1
            cprob = 0
            for a in range(action_num):
                cprob = cprob + policy[a]
                action = a
                if random*10 < cprob:
                    break

            #action = np.random.choice(env.action_space.n, p=policy)

            return action

    @classmethod
    def Qupdate(self, Q, new_state, state, action, reward):
        """ Q$BCM$N99?7(B """
        return (Q[int(state[0]), int(state[1]), action] + alpha*(reward-Q[int(state[0]), int(state[1]), action]+gamma*np.max(Q[int(new_state[0]), int(new_state[1]),:])))

if __name__=='__main__':
    Ql = Qlearning
    Q = np.zeros([state_num, state_num, action_num])
    eps = 0.8
    new_state = np.zeros(2)
    reward_observation = np.zeros(2)

    env = gym.make('MountainCar-v0')
    #env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / state_num

    for i_episode in range(epiode_num):
        state = env.reset()
        eps *= 0.999
        if eps < 4e-2:
            eps = 0.04
        print("Episode {}".format(i_episode+1))
        for t in range(step):
            # $B9TF0(Ba$B$N3MF@(B
            # action = env.action_space.sample()
            action = Ql.choose_action(Q, state, eps)

            # $B<B9T(B
            observation, reward, done, _ = env.step(action)

            # reward $B$N3MF@(B
            # reward = 0.01 / (0.01+(env_high[1]-abs(new_state[1]))**2)

            # $BN%;62=(B
            # new_state[0] = round(new_state[0]*100)+120
            # new_state[1] = round(new_state[1]*100)+70
            new_state[0] = int((observation[0] - env_low[0])/env_dx[0])
            new_state[1] = int((observation[1] - env_low[1])/env_dx[1])
            
            # reward $B$N3MF@(B
            reward_observation[0] = new_state[0]
            reward_observation[1] = int((abs(observation[1]) - env_low[1])/env_dx[1])
            state_ave = np.sum(reward_observation)/2
            reward = 10 / (10+(state_num-state_ave)**2)

            # Q$B3X=,$N99?7(B
            Q[int(state[0]), int(state[1]), action] = Ql.Qupdate(Q, new_state, state, action, reward)

            #if np.max(Q) < 0 and t > 100:
            #    print("ERROR: Q=0")
            #    sys.exit()

            # $B>uBV$H9TF0$N5-O?(B
            state = new_state

            if done:
                print eps
                if new_state[0] == state_num-10:
                    goal_total += 1
                    if i_episode > epiode_num-101:
                        goal += 1

                print("Episode finished after {} timesteps".format(t+1))
                break

    run_episode(env, Q, True)

    print("Q(s,a) =\n{}".format(Q))
    print("Goal total: {}".format(goal_total))
    print("{}/100".format(goal))
    print(eps)

    env.close()
    #gym.upload('/tmp/cartpole-experiment-1', api_key='sk_H98pDlIsRJWR9A3QmoKHhQ')
