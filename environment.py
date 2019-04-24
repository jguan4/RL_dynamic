import sys
sys.dont_write_bytecode = True

import gym
from env import Henon
from env import Lorenz
from env import Henon_Net
from env import Net
import env
import numpy as np
import random
from PIL import Image
import utils
import time

class Henon_Map:

    def __init__(self, action_range, hs, delay, type="Henon", history_pick=1, period=1):
        self.name = type + str(time.time())
        self.env = Henon(delay=delay,period=period)
        self.period = period
        if delay:
            self.state_dimension = [max(2,period)]
        else: self.state_dimension = [2]
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.state_dimension)
        self.action_range = action_range
        self.hs = hs
        self.action_space = np.multiply(self.hs, self.action_range)
        self.action_space_size = self.action_space.size 
        self.state_shape = [None, self.history_pick*self.state_dimension[0]] 
        self.action_shape = [None, self.history_pick*1] 
        self.history = []

    # returns a random action
    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action):
        return self.action_space[action]

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
            info['true_done'] = done
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

    def record_traj(self):
        traj = self.env.x_traj
        x_bar = self.env.x_bars
        return traj, x_bar

    def __str__(self):
        return self.name + '\nactions: {0}\n period: {1}'.format(self.action_space, self.period)

##################################################################
##################################################################

class Henon_Network:

    def __init__(self, num_act, max_mag, act_type, num_n, obs, delay, type="Henon_Network", history_pick=1, period=1):
        self.name = type + str(time.time())
        self.dim = 2
        self.obs = obs
        self.net = Net(num_n = num_n, dim = self.dim, obs = obs)
        self.action_space = self.net.create_action_range(num_act,max_mag,act_type)
        self.action_ref = self.net.create_line_action(self.action_space)
        self.env = Henon_Net(net = self.net, delay=delay, period=period)
        self.period = period
        self.history_pick = history_pick
        obs_num = len(obs)
        if delay:
            self.state_dimension = [max(self.dim*num_n,period)]
            if obs_num == 1:
                self.state_shape = [None, self.history_pick*self.state_dimension[0]]
            else: 
                self.state_shape = [self.state_dimension[0],obs_num]
        else: 
            self.state_dimension = [period*obs_num] # change later
            self.state_shape = [None, self.history_pick*self.state_dimension[0]] 
        self.state_space_size = history_pick * np.prod(self.state_dimension)
        self.action_shape = [None, self.history_pick*1] 
        self.action_space_size = np.shape(self.action_ref)[0]
        self.history = []

    # returns a random action
    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action):
        return self.action_ref[action]

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
            info['true_done'] = done
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
        if not self.state_shape[0]:
            result = np.reshape(result,(self.history_pick*self.state_dimension[0]))
        return result

    def add_history(self, state):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        # temp = utils.process_image(state, detect_edges=self.detect_edges, flip=self.flip_episode)
        temp = state
        self.history.append(temp)

    def record_traj(self):
        traj = self.env.x_traj
        x_bar = self.env.x_bars
        return traj, x_bar

    def __str__(self):
        return self.name + '\nactions: {0}\n period: {1}\n observed: {2}\n '.format(self.action_space, self.period, self.obs)

##################################################################
##################################################################

class Lorenz_Attractor:

    def __init__(self, action_range, hs, delay, type="Lorenz", history_pick=1, period=1):
        self.name = type + str(time.time())
        self.env = Lorenz(delay=delay,period=period)
        self.period = period
        if delay:
            self.state_dimension = [max(4,period)]
        else: self.state_dimension = [3]
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.state_dimension)
        self.action_range = action_range
        self.hs = hs
        self.action_space = np.multiply(self.hs, self.action_range)
        self.action_space_size = self.action_space.size 
        self.state_shape = [None, self.history_pick*self.state_dimension[0]] 
        self.action_shape = [None, self.history_pick*1] 
        self.history = []

    # returns a random action
    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action):
        return self.action_space[action]

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
            info['true_done'] = done
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

    def record_traj(self):
        traj = self.env.x_traj
        x_bar = self.env.x_bars
        return traj, x_bar

    def __str__(self):
        return self.name + '\nactions: {0}\n period: {1}'.format(self.action_space, self.period)


##################################################################
##################################################################


class Acrobot:
    # C:\Users\jiaji\AppData\Local\lxss\rootfs\usr\local\lib\python3.6\dist-packages\gym\envs
    def __init__(self, type="Acrobot_mag", history_pick=1, factor=1, normalize=97*np.square(np.pi),mag=1):
        self.name = type + str(time.time())
        self.env = gym.make(type + '-v1', factor)
        self.env.factor = factor
        self.env.mag = mag
        self.factor = factor
        self.normalize = normalize
        self.state_dimension = [6]
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.state_dimension)
        self.action_space_size = 3
        self.state_shape = [None, self.history_pick*self.state_dimension[0]] 
        self.history = []
        # self.action_dict = {0:0, 1:1, 2:2, 3:3, 4:4}
        self.action_dict = {0:0, 1:1, 2:2}
        self.link1 = 1
        self.link2 = 1

    # returns a random action
    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action):
        return action

    # resets the environment and returns the initial state
    def reset(self, test=False):
        return self.process(self.env.reset())

    # take action 
    def step(self, action, test=False):
        action = self.map_action(action)
        total_reward = 0
        # n = self.factor
        n = 1
        for i in range(n):
            next_state, reward, done, info = self.env.step(action)
            reward, done = self.analyze_state(self.process(next_state))
            total_reward += reward
            info = {'true_done': done}
            if done: break
        total_reward = total_reward/(i+1)
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
        # reward = height_ave[0] - ((np.square(theta1dot_ave)+np.square(theta2dot_ave))/(self.normalize))
        done = True if height_ave[0]< 1 else False
        reward = 0 if done else 1
        # reward = height_ave[0]
        return reward, done

    def __str__(self):
        return self.name + '\nseed: {0}\nactions: {1}'.format(0, self.action_dict)


class Pendulum:

    def __init__(self, type="Pendulum", history_pick=1, factor=1,normalize=1):
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
