import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import random
from PIL import Image
import utils
import time

class Acrobot:

    # Parameters
    # - type: Name of environment. Default is classic Car Racing game, but can be changed to introduce perturbations in environment
    # - history_pick: Size of history
    # - seed: List of seeds to sample from during training. Default is none (random games)
    def __init__(self, type="Acrobot", history_pick=4):
        self.name = type + str(time.time())
        self.env = gym.make(type + '-v1')
        self.state_dimension = [1,6]
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.state_dimension)
        self.action_space_size = 3
        self.state_shape = [None, self.history_pick] + list(self.state_dimension)
        self.history = []
        self.action_dict = {0: [-1], 1: [0], 2: [1]}
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
        action = self.map_action(action)
        total_reward = 0
        n = 1
        for i in range(n):
            next_state, reward, done, info = self.env.step(action)
            reward = self.analyze_state(next_state):
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
            result = np.concatenate((result, np.array(self.history)))
        else:
            result = np.array(self.history)
        return result

    def add_history(self, state):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        # temp = utils.process_image(state, detect_edges=self.detect_edges, flip=self.flip_episode)
        temp = state
        self.history.append(temp)

    def analyze_state(self, state):
        p1 = [-self.link1*state[:,0], self.link2*state[:,1]]
        p2 = [p1[:,0]-self.link2*(state[:,0]*state[:,2]-state[:,1]*state[:,3]), 
            p1[:,1]+self.link2*(state[:,0]*state[:,1]+state[:,2]*state[:,3])]
        height_std = np.std(p2, axis=0)
        height_ave = np.average(p2, axis=0)
        reward = self.reward_func(height_ave, height_std)
        return reward

    def reward_func(self, height_ave, height_std):
        return -height_std[0]

    def __str__(self):
        return self.name + '\nseed: {0}\nactions: {1}'.format(0, self.action_dict)
