import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import random
from PIL import Image
import utils
import time

class Acrobot:
    def __init__(self, type="Acrobot", history_pick=4, factor=1):
        self.name = type + str(time.time())
        self.env = gym.make(type + '-v1')
        self.env.factor = factor
        print(self.env.AVAIL_TORQUE)
        self.state_dimension = [6]
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.state_dimension)
        self.action_space_size = 3
        self.state_shape = [None, self.history_pick*self.state_dimension[0]] 
        self.history = []
        self.action_dict = {0: -1, 1: 0, 2: 1}
        self.link1 = 1
        self.link2 = 1

    # returns a random action
    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action):
        return self.action_dict[action]

    # resets the environment and returns the initial state
    def reset(self, test=False):
        return self.process(self.env.reset())

    # take action 
    def step(self, action, test=False):
        # action = self.map_action(action)
        total_reward = 0
        n = 1
        for i in range(n):
            next_state, reward, done, info = self.env.step(action)
            reward = self.analyze_state(self.process(next_state))
            total_reward += reward
            info = {'true_done': done}
            if done: break
        processed_next_state = self.process(next_state)    
        return processed_next_state, total_reward, done, info

    def render(self):
        self.env.render()

    # process state and return the current history
    def process(self, state):
        self.add_history(state)
        if len(self.history) < self.history_pick:
            zeros = np.zeros(self.state_dimension)
            result = np.tile(zeros, ((self.history_pick - len(self.history)), 1))
            result = np.concatenate((result, np.array(self.history)),axis=0)
        else:
            result = np.array(self.history)
        result = np.reshape(result,(self.history_pick*self.state_dimension[0]))
        return result

    def add_history(self, state):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        # temp = utils.process_image(state, detect_edges=self.detect_edges, flip=self.flip_episode)
        temp = state
        self.history.append(temp)

    def analyze_state(self, state):
        costheta1 = np.array(state[0::self.state_dimension[0]])
        sintheta1 = np.array(state[1::self.state_dimension[0]])
        costheta2 = np.array(state[2::self.state_dimension[0]])
        sintheta2 = np.array(state[3::self.state_dimension[0]])
        theta1dot = np.array(state[4::self.state_dimension[0]])
        theta2dot = np.array(state[5::self.state_dimension[0]])
        p1 = np.array([-self.link1*costheta1, self.link2*sintheta1])
        p2 = np.array([p1[0]-self.link2*(costheta1*costheta2-sintheta1*sintheta2), 
            p1[1]+self.link2*(costheta1*sintheta1+costheta2*sintheta2)])
        height_std = np.std(p2, axis=1)
        height_ave = np.average(p2, axis=1)
        theta1dot_ave = np.average(theta1dot)
        theta2dot_ave = np.average(theta2dot)
        reward = (np.square(height_ave[0])) - (0.1*(np.square(theta1dot_ave)+np.square(theta2dot_ave)))
        return reward

    def __str__(self):
        return self.name + '\nseed: {0}\nactions: {1}'.format(0, self.action_dict)


class Pendulum:

    def __init__(self, type="Pendulum", history_pick=4):
        self.name = type + str(time.time())
        self.env = gym.make(type + '-v0')
        self.state_dimension = [3]
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.state_dimension)
        self.action_space_size = 5
        self.state_shape = [None, self.history_pick*self.state_dimension[0]] 
        self.history = []
        self.action_dict = {0: [-2], 1: [-1], 2: [0], 3: [1], 4: [2]}

    # returns a random action
    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action):
        return self.action_dict[action]

    # resets the environment and returns the initial state
    def reset(self, test=False):
        return self.process(self.env.reset())

    # take action 
    def step(self, action, test=False):
        action = self.map_action(action)
        total_reward = 0
        n = 1
        for i in range(n):
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            info = {'true_done': done}
            if done: break
        processed_next_state = self.process(next_state)    
        return processed_next_state, total_reward, done, info

    def render(self):
        self.env.render()

    # process state and return the current history
    def process(self, state):
        self.add_history(state)
        if len(self.history) < self.history_pick:
            zeros = np.zeros(self.state_dimension)
            result = np.tile(zeros, ((self.history_pick - len(self.history)), 1))
            result = np.concatenate((result, np.array(self.history)),axis=0)
        else:
            result = np.array(self.history)
        result = np.reshape(result,(self.history_pick*self.state_dimension[0]))
        return result

    def add_history(self, state):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        # temp = utils.process_image(state, detect_edges=self.detect_edges, flip=self.flip_episode)
        temp = state
        self.history.append(temp)

    def __str__(self):
        return self.name + '\nseed: {0}\nactions: {1}'.format(0, self.action_dict)
