import argparse
import gym
from collections import deque
from DQNAgent import DQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue
from Environment import *

if __name__ == '__main__':
    
    train_model = "save/DQNParams_600.h5"
    play_episodes = 30000

    env = Environment()
    agent = DQNAgent(epsilon=0) # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load(train_model)

    for e in range(play_episodes):
        init_state = env.reset("DQN")
        init_state = process_state_image(init_state)

        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        
        while True:
            env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            next_state, reward, done, info = env.step(action,"DQN")

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)

            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(e+1, play_episodes, time_frame_counter, float(total_reward)))
                break
            time_frame_counter += 1
