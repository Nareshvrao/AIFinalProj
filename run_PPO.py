import argparse
from Environment import *
from PPOAgent import *


if __name__ == "__main__":
    agent = PPOAgent()
    env = Environment()
    agent.load_param()
    state = env.reset("PPO")

    for i_ep in range(10):
        score = 0
        state = env.reset("PPO")

        for t in range(100000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]), "PPO")
           
            env.render()
            score += reward
            state = state_
            if done or die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
