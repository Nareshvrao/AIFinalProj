
import gym
import numpy as np

class Environment():
     def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(1000)
        self.reward_threshold = self.env.spec.reward_threshold

     def reset(self,typee):
          if (typee == "DQN"):
            return self.env.reset()
          else:
            self.counter = 0
            self.av_r = self.reward_memory()

            self.die = False
            img_rgb = self.env.reset()
            img_gray = self.rgb2gray(img_rgb)
            self.stack = [img_gray] * 4
            return np.array(self.stack)

     def step(self, action,typee):
        if (typee == "DQN"):
            return self.env.step(action)
        else:
            total_reward = 0
            for i in range(8):
                  img_rgb, reward, die, _ = self.env.step(action)
                  # don't penalize "die state"
                  if die:
                     reward += 100
                  # green penalty
                  if np.mean(img_rgb[:, :, 1]) > 185.0:
                     reward -= 0.05
                  total_reward += reward
                  # if no reward recently, end the episode
                  done = True if self.av_r(reward) <= -0.1 else False
                  if done or die:
                     break
            img_gray = self.rgb2gray(img_rgb)
            self.stack.pop(0)
            self.stack.append(img_gray)
            assert len(self.stack) == 4
            return np.array(self.stack), total_reward, done, die

     def render(self, *arg):
        self.env.render(*arg)

     @staticmethod
     def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

     @staticmethod
     def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory