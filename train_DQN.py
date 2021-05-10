import argparse
import gym
from collections import deque
from DQNAgent import DQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue
from Environment import *

if __name__ == '__main__':
  
    env = Environment()
    agent = DQNAgent()
    

    for e in range(4):
        init_state = env.reset("DQN")
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False
        
        while True:
           
            env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            reward = 0
            for _ in range(3):
                next_state, r, done, info = env.step(action,"DQN")
                reward += r
                if done:
                    break

            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
                break
            time_frame_counter += 1
           
        agent.update_target_model()

        
    agent.save('./save/DQNParams_{}.h5'.format(e))

   
