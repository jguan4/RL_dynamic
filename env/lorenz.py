import numpy as np
import utils
from numpy import linalg as LA

class Lorenz:
	def __init__(self, delay=False, period=1):
		# initializer
		self.state = None
		self.t = None
		self.x_traj = None
		self.o_traj = None
		self.in_neigh = False
		self.consecutive_reward = None
		self.delay = delay

		# parameters
		self.period = period
		self.substep = 2
		self.h = 0.025
		self.dt = self.period*self.h*self.substep
		self.radius = 0.05
		self.past = 10
		self.terminate = 0.02
		self.direction = [1,1,0]
		if self.delay:
			self.dim = max(4,self.period)
		else: self.dim = self.period
		self.x_bars = np.empty((0,3),float)
		# self.x_bar = None

	def reset(self):
		# setting up first delay coordinates
		init_state =  [1, 0, 0] + np.random.normal(0, 0.1, 3)
		self.x_traj = [init_state]
		self.t = 0
		act = np.multiply(0, self.direction)
		ns_p = self.lor63(self.x_traj[-1], act)
		self.t = self.t + self.dt

		# first delay coordinates obtained
		# update trajectories
		if self.delay:
			self.state = ns_p[:,0]
		else: self.state = ns_p[-1]
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
		
		if self.period>1:
			traj_dev = np.absolute(traj[-1]-traj[-2])
			norm_dist = LA.norm(traj_dev,2)
			# norm_dist = traj_dev
			reward = -norm_dist
			
			# mid_dev = np.absolute(traj[-1]-np.flip(traj[-2],0))
			# mid_dev = np.absolute(traj[-1][1::]-traj[-2][-(self.period-1)::])
			# mid_norm_dist = LA.norm(mid_dev,2)

			# this is to test the repeatedness of first entry in the state:
			minus_vec = np.absolute(traj[-1][1::]-traj[-1][0])
			min_minus_vec = np.min(minus_vec)
			if norm_dist<self.radius and min_minus_vec<0.15:
				reward-=1
		else:
			# traj_dev = np.absolute(traj[-1]-np.flip(traj[-2],0)) # for period 1 only
			traj_dev = np.absolute(traj[-1]-traj[-2])
			norm_dist = LA.norm(traj_dev,2)
			reward = -norm_dist
			
		# self.update_radius()
		# if not np.any(self.x_bars):
		# traj_dev = np.absolute(traj[-1]-np.flip(traj[-2],0)) # for period 1 only
		# else:
		# 	x_bar = np.mean(self.x_bars, axis=0)
		# 	traj_dev = np.absolute(traj[-1]-x_bar)

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
		if self.t>50:
			ter = True
		return (reward,ter,info)

	def render(self):
		return None

	def step(self, a):
		act = np.multiply(a, self.direction)
		ns_p = self.lor63(self.x_traj[-1], act)
		self.t = self.t + self.dt
		# self.state = ns_p[-1]
		if self.delay:
			self.state = ns_p[:,0]
		else: self.state = ns_p[-1]

		# only for producing trajectory, not for reference use
		self.x_traj = np.append(self.x_traj,ns_p,axis=0)
		self.o_traj = np.append(self.o_traj,[self.state],axis=0)
		(reward,terminal,info) = self._terminal()

		return (self.state, reward, terminal, info)

	def lor63(self, w, act):
		y = np.zeros((self.dim,3))
		w = w + act
		for i in range(self.dim):
			y[i,:] = self.rk4step(w)
			w = y[i,:].copy()
		y = np.array(y)
		return y

	def rk4step(self, w):
		t = self.t
		h = self.h
		# one step of the Runge-Kutta order 4 method
		for i in range(self.substep):
			s1 = self.ydot(t,w);
			s2 = self.ydot(t+h/2, w+h*s1/2);
			s3 = self.ydot(t+h/2, w+h*s2/2);
			s4 = self.ydot(t+h, w+h*s3);
			w = w + h*(s1+2*s2+2*s3+s4)/6;
		return w

	def ydot(self,t,y):
		s = 10
		r = 28 
		b = 8/3
		z = np.zeros(3)
		z[0] = -s*y[0] + s*y[1]
		z[1] = -y[0]*y[2] + r*y[0] - y[1]
		z[2] = y[0]*y[1] - b*y[2]
		return z

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