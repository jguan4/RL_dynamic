import numpy as np
import math
import utils
from numpy import linalg as LA

class Henon_Net:
	def __init__(self, net, delay=True, period=1):
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
		self.radius = np.power(0.1,2) #0.05
		self.past = 10
		self.terminate = np.power(0.05,2)
		self.dim = 2

		# parameters for network and henon
		self.net = net
		self.adj = self.net.adj
		self.obs_arr = self.net.obs_arr
		self.num_n = self.net.num_n
		self.obs_num = self.net.obs_num
		self.obs = self.net.obs
		np.random.seed(10)
		self.p1 = [3,-2,4] #[3, -2] #[3.23139619298002,3.00622558480782] #3 + 0.1*np.random.rand(self.num_n)
		self.p2 = [-0.5,-0.7,-0.6] #[-0.5, -0.7] #[-0.209905529522117,-0.175358835238416] #-0.4 + 0.1*np.random.rand(self.num_n)
		self.p3 = [0.7,0.8,0.5] #[0.7, 0.8]
		self.cw = [0.5, 0.3,0.3] #[0.5, 0.3]
		if self.delay: 
			self.iter_step = 9#max(self.dim*self.num_n,self.period)
			# self.iter_step = math.ceil(self.iter_step/self.obs_num)
		else: self.iter_step = self.period
		self.dt = self.iter_step
		self.x_bars = np.empty((0,(self.num_n*self.dim)*2),float)
		# self.x_bar = [-3.74253810363669,-3.72571485279983,-3.74253810363669,-3.72571485279983]
		# self.x_bar = [1.28412153700697,1.27988461040069,1.28412153700697,1.27988461040069]


	def reset(self):
		# setting up first delay coordinates
		init_state =  np.random.rand(self.num_n*self.dim)*2-1 #[1.4698,1.4923,1.4698,1.4923]#
		self.x_traj = [init_state]
		self.t = 0
		act = np.zeros(self.num_n*self.dim)
		self.t = self.t + self.dt
		ns_p = self.henon_net(self.t, self.x_traj[-1], act)

		# first delay coordinates obtained
		# update trajectories
		if self.delay:
			self.state = np.squeeze(ns_p[:,self.obs])
			self.state = np.reshape(self.state,(self.iter_step*self.obs_num))
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
	def _terminal(self,a):
		traj = self.o_traj
		ter = False
		info = {}
		
		if self.delay:
			if self.period>1:
				traj_dev = np.absolute(traj[-1]-traj[-2])
				norm_dist = LA.norm(traj_dev)
				minus_vec = np.absolute(traj[-1][1::]-traj[-1][0])
				min_minus_vec = np.min(minus_vec)
				if norm_dist<self.radius and min_minus_vec<0.15:
					norm_dist+=1
			else:
				# traj_dev = np.absolute(traj[-1]-np.flip(traj[-2],0)) 
				traj_dev = np.absolute(traj[-1]-traj[-2]) 
				norm_dist = LA.norm(traj_dev)
		else:
			traj_dev = np.absolute(traj[-1]-traj[-2])
			# traj_dev = np.absolute(traj[-1]-self.x_bar)
			# norm_dist = LA.norm(traj_dev,2)
			norm_dist = np.power(LA.norm(traj_dev,2),2)

		reward = -norm_dist

		info['Fixed_Point'] = None
		if np.absolute(reward)<self.radius:
			c_x = self.x_traj[-1-self.period]
			self.x_bars = np.append(self.x_bars,[np.append(c_x,a)],axis=0)
			info['Fixed_Point'] = self.state
			# if np.absolute(reward)<self.terminate:
			# 	ter = True

		if self.t/self.dt > 1e3:
			ter = True
		# if self.state[-1][0]>100:
		# 	ter = True
		# 	reward = -10
		return (reward,ter,info)

	def render(self):
		return None

	def step(self, a):
		self.t = self.t + self.dt
		ns_p = self.henon_net(self.t, self.x_traj[-1], a)
		if self.delay:
			self.state = np.squeeze(ns_p[:,self.obs])
			self.state = np.reshape(self.state,(self.iter_step*self.obs_num))
		else: 
			self.state = ns_p[-1]
			self.state = self.state[self.obs]

		# only for producing trajectory, not for reference use
		self.x_traj = np.append(self.x_traj,ns_p,axis=0)
		self.o_traj = np.append(self.o_traj,[self.state],axis=0)
		(reward,terminal,info) = self._terminal(a)

		return (self.state, reward, terminal, info)

	def henon_net(self,t,w,act):
		y = np.zeros((self.iter_step,self.dim*self.num_n))
		# y[0] = -1.4*np.square(w[0])+w[1]+1
		# y[1] = 0.3*w[0]
		w = w + act
		for i in range(self.iter_step):
			p_x1 = w[:self.num_n:]
			p_x2 = w[self.num_n::]
			y[i,:self.num_n:] = np.multiply(self.p1,np.cos(p_x1))+np.multiply(self.p2,p_x2)+np.multiply(self.cw,np.matmul(p_x1,np.transpose(self.adj)))
			y[i,self.num_n::] = np.multiply(self.p3,p_x1)
			w = y[i,:].copy()
		y = np.array(y)
		return y