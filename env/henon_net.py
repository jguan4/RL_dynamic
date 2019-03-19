import numpy as np
import utils
from numpy import linalg as LA
from .net import *

class Henon_Net:
	def __init__(self, num_n, obs, delay=True, period=1):
		# initializer
		self.state = None
		self.t = None
		self.x_traj = None
		self.o_traj = None
		self.in_neigh = False
		self.consecutive_reward = None
		self.delay = delay

		# parameters for environment
		self.period = period
		self.radius = 0.05
		self.past = 10
		self.terminate = 0.02
		self.dim = 2

		# parameters for network and henon
		self.num_n = num_n
		self.obs = obs
		self.adj = create_full_adj(self.num_n)
		self.obs_arr = create_obs_arr(self.num_n,
			self.dim,self.obs)
		obs_num = len(self.obs)
		self.direction = [1,1,0,0]
		self.p1 = 2 + 0.1*np.random.rand(self.num_n)
		self.p2 = 0.4 + 0.1*np.random.rand(self.num_n)

		# self.x_bar = [0.6314,0.1894]
		# self.x_bar = [1.2019, 1.2019]
		# self.x_bar = [0.8385, 0.8385]
		if self.delay:
			self.iter_step = max(2,self.period)
		else: self.iter_step = self.period
		self.dt = self.iter_step
		self.x_bars = np.empty((0,self.num_n*self.dim),float)
		self.x_bar = None

	def reset(self):
		# setting up first delay coordinates
		init_state =  np.random.rand(self.num_n*self.dim)*2-1
		self.x_traj = [init_state]
		self.t = 0
		act = np.multiply(0, self.direction)
		self.t = self.t + self.dt
		ns_p = self.henon_net(self.t, self.x_traj[-1], act)

		# first delay coordinates obtained
		# update trajectories
		if self.delay:
			self.state = np.matmul(ns_p,self.obs_arr)
		else: 
			self.state = ns_p[-1]
			self.state = self.state[self.obs]

		self.x_traj = np.append(self.x_traj,ns_p,axis=0)
		self.o_traj = [self.state]

		return self.state

	# cat: 0 not stationary - reward 0
	#      1 stationary - reward 1
	#      2 went out of neighborhood - reward -1
	#      3 close to the fixed point, terminate
	def _terminal(self):
		traj = self.o_traj
		ter = False
		info = {}
		
		if self.delay:
			if self.period>1:
				traj_dev = np.absolute(traj[-1]-traj[-2])
				norm_dist = LA.norm(traj_dev,2)
				minus_vec = np.absolute(traj[-1][1::]-traj[-1][0])
				min_minus_vec = np.min(minus_vec)
				if norm_dist<self.radius and min_minus_vec<0.15:
					norm_dist+=1
			else:
				traj_dev = np.absolute(traj[-1]-np.flip(traj[-2],0)) 
				norm_dist = LA.norm(traj_dev,2)
		else:
			traj_dev = np.absolute(traj[-1]-traj[-2])
			norm_dist = LA.norm(traj_dev,2)

		reward = -norm_dist

		# 	# self.update_radius()
		# 	if not np.any(self.x_bars):
		# 		traj_dev = np.absolute(traj[-1]-traj[-2]) # for period 1 only
		# 	else:
		# 		x_bar = np.mean(self.x_bars, axis=0)
		# 		traj_dev = np.absolute(traj[-1]-x_bar)
		# 	# traj_dev = np.absolute(traj[-1]-np.flip(traj[-2],0)) # for period 1 only	

		# if self.period>1:
		# 	past_dev = LA.norm(np.absolute(traj[-1]-traj[-2]),2)
		# else:
		# 	past_dev = self.radius + 1

		# if norm_dist<self.radius and past_dev > self.radius*1.5:

		if np.absolute(reward)<self.radius:
			c_x = self.x_traj[-1]
			self.x_bars = np.append(self.x_bars,[c_x],axis=0)
			info['Fixed_Point'] = self.state
			# if self.in_neigh: self.consecutive_reward += 1
			# self.in_neigh = True
			# if LA.norm(np.absolute(traj[-1]-self.x_bar))<self.terminate:
			# if LA.norm(np.absolute(traj[-1]-x_bar))<self.terminate:
			# if norm_dist<self.terminate:
			# 	ter = True
		else:
			info['Fixed_Point'] = None
			# if self.in_neigh and cat == 0:
				# if self.radius < 0.025: self.radius = self.radius*2
				# self.hs = self.hs*2.
				# self.action_space = np.multiply(self.hs, [+1.0, 0., -1.0])
		if self.t/self.dt > 1000:
			ter = True
		if traj[-1][0]>10:
			ter = True
			reward = -10
		return (reward,ter,info)

	def render(self):
		return None

	def step(self, a):
		act = np.multiply(a, self.direction)
		self.t = self.t + self.dt
		ns_p = self.henon_net(self.t, self.x_traj[-1], act)
		# self.state = ns_p[-1]
		if self.delay:
			self.state = np.matmul(ns_p,self.obs_arr)
		else: 
			self.state = ns_p[-1]
			self.state = self.state[self.obs]

		# only for producing trajectory, not for reference use
		self.x_traj = np.append(self.x_traj,ns_p,axis=0)
		self.o_traj = np.append(self.o_traj,[self.state],axis=0)
		(reward,terminal,info) = self._terminal()

		return (self.state, reward, terminal, info)

	def henon_net(self,t,w,act):
		y = np.zeros((self.iter_step,self.dim*self.num_n))
		# y[0] = -1.4*np.square(w[0])+w[1]+1
		# y[1] = 0.3*w[0]
		w = w + act
		for i in range(self.iter_step):
			p_x1 = w[:self.num_n:]
			p_x2 = w[self.num_n::]
			y[i,:self.num_n:] = np.multiply(self.p1,np.cos(p_x1))+np.multiply(self.p2,p_x2)+np.matmul(p_x1,np.transpose(self.adj))/self.num_n
			y[i,self.num_n::] = p_x1
			w = y[i,:].copy()
		y = np.array(y)
		return y

	def update_radius(self):
		if self.t>self.past:
			# stay within 2*previous radius for 10 steps
			traj_min = np.amin(self.x_traj[-1-self.past::],axis=0)
			traj_max = np.amax(self.x_traj[-1-self.past::],axis=0)
			traj_dev = np.absolute(traj_max-traj_min)
			if traj_dev[0]<self.radius and traj_dev[1]<self.radius:
				self.radius = self.radius/2.
				# self.hs = self.hs/2.
				# self.action_space = np.multiply(self.hs, [+1.0, 0., -1.0])