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
		self.action_space = np.multiply(hs, [+1.0, 0., -1.0])

	def reset(self):
		self.state = [-0.2, 0.15] + np.random.normal(0, 0.1, 2)
		self.t = 0
		self.x_traj = [self.state]
		return self.state

	def _terminal(self):

		# ret = False
		# cat = 0
		# if s_abs[0] > 1.5 or s_abs[1] > 0.4:
		# 	ret = True
		# 	cat = 1
		# elif s_dev[0]<0.025 and s_dev[1]<0.025:
		# 	ret = True
		# 	cat = 2
		# elif self.t > 500:
		# 	ret = True

		traj = self.x_traj
		s = self.state
		s_abs = np.absolute(s)
		ret = False
		cat = 0 
		# cat: 0 not finished
		#      1 out of bound
		#      2 near fixed point
		#      3 overtime
		if np.any(self.x_bar):
			s_dev = np.absolute(s-self.x_bar)
			if s_dev[0]<0.025 and s_dev[1]<0.025:
				ret = True
				cat = 2
				self.x_bar = self.state
		else:
			traj_dev = np.absolute(traj[-1]-traj[-2])
			if traj_dev[0]<0.025 and traj_dev[1]<0.025:
				ret = True
				cat = 2
				self.x_bar = self.state

		if s_abs[0]> 1e3 or s_abs[1]>1e3:
			ret = True
			cat = 1
		elif self.t>500:
			ret = True
			cat = 3
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
			reward = -1. 
			info['Fixed_Point'] = None
		elif cat == 1:
			reward = -100.
			info['Fixed_Point'] = 'Out_of_bound'
		elif cat == 2: 
			reward = 0.
			info['Fixed_Point'] = self.state
		else: 
			reward = -1.
			info['Fixed_Point'] = 'Overtime'
		return (self.state, reward, terminal, info)

	def henon(self,t,w):
		y = np.zeros(2)
		y[0] = -1.4*np.square(w[0])+w[1]+1
		y[1] = 0.3*w[0]
		return y