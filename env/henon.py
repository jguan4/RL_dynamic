import numpy as np

class Henon:
	def __init__(self, hs = 0.1):
		self.substep = 1
		self.state = None
		self.t = None
		self.dt = 1
		# self.x_bar = [0.6314,0.1894]
		self.x_bar = None
		self.x_traj = None
		self.radius = 0.025
		self.past = 10
		self.hs = hs
		self.action_space = np.multiply(self.hs, [+1.0, 0., -1.0])

	def reset(self):
		self.state = [-0.2, 0.15] + np.random.normal(0, 0.1, 2)
		self.t = 0
		self.x_traj = [self.state]
		return self.state

	def _terminal(self):
		traj = self.x_traj
		cat = 0 
		# cat: 0 not stationary - reward 0
		#      1 stationary - reward 1*self.radius/0.025
		ret = False
		traj_dev = np.absolute(traj[-1]-traj[-2])
		self.update_radius()
		if traj_dev[0]<self.radius and traj_dev[1]<self.radius:
			cat = 1
			self.x_bar = self.state
		return (ret, cat)

	def render(self):
		return None

	def step(self, a):
		action = self.action_space[a]
		s_aft = self.state + [action, 0]
		self.t = self.t + self.dt
		ns = self.henon(self.t, s_aft)
		self.state = ns
		self.x_traj = np.append(self.x_traj,[self.state],axis=0)
		(terminal, cat) = self._terminal()
		info = {}
		if cat==0:
			reward = 0. 
			info['Fixed_Point'] = None
		elif cat == 1:
			reward = self.radius/0.025
			info['Fixed_Point'] = self.state
		return (self.state, reward, terminal, info)

	def henon(self,t,w):
		y = np.zeros(2)
		# y[0] = -1.4*np.square(w[0])+w[1]+1
		# y[1] = 0.3*w[0]
		y[0] = 2*np.cos(w[0])+0.4*w[1]
		y[1] = w[0]
		return y

	def update_radius(self):
		if self.t>self.past:
			# stay within 2*previous radius for 10 steps
			traj_min = np.amin(self.x_traj[-1-self.past::],axis=0)
			traj_max = np.amax(self.x_traj[-1-self.past::],axis=0)
			traj_dev = np.absolute(traj_max-traj_min)
			if traj_dev[0]<2*self.radius and traj_dev[1]<2*self.radius:
				self.radius = self.radius/2.
				self.hs = self.hs/2.
				self.action_space = np.multiply(self.hs, [+1.0, 0., -1.0])