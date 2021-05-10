import argparse

from utils import DrawLine
import matplotlib
from Environment import *
from PPOAgent import *
import matplotlib.pyplot as plt


if __name__ == "__main__":
    agent = PPOAgent()
    env = Environment()
    draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")
    moving_scores = []
    moving_score = 0
    state = env.reset("PPO")
    for i_ep in range(1000):
        score = 0
        state = env.reset("PPO")

        for t in range(10000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]),"PPO")
            
            env.render()
            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            
            state = state_
            if done or die:
                break
        moving_score = moving_score * 0.99 + score * 0.01
        moving_scores.append(moving_score)

    agent.save_param()
