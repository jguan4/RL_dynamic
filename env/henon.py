import numpy as np

class Henon:
	def __init__(self, hs = 0.1):
		self.substep = 1
		self.state = None
		self.t = None
		self.dt = 1
		self.x_bar = [0.6314,0.1894]
		self.action_space = np.multiply(hs,[+1.0, 0., -1.0])

	def reset(self):
		self.state = [-0.2, 0.15] + np.random.normal(0, 0.1, 2)
		self.t = 0
		return self.state

	def _terminal(self):
		s = self.state
		s_abs = np.absolute(s)
		s_dev = np.absolute(s-self.x_bar)
		ret = False
		cat = 0
		if s_abs[0] > 1.5 or s_abs[1] > 0.4:
			ret = True
			cat = 1
		elif s_dev[0]<0.025 and s_dev[1]<0.025:
			ret = True
			cat = 2
		elif self.t > 500:
			ret = True
		return (ret, cat)

	def render(self):
		return None

	def step(self, a):
		action = self.action_space[a]
		s_aft = self.state + [action, 0]
		self.t = self.t + self.dt
		ns = self.henon(self.t, s_aft)
		self.state = ns
		(terminal, cat) = self._terminal()
		if cat==0:
			reward = -1. 
		elif cat == 1:
			reward = -100.
		else: reward = 0.
		return (self.state, reward, terminal, {})

	def henon(self,t,w):
		y = np.zeros(2)
		y[0] = -1.4*np.square(w[0])+w[1]+1
		y[1] = 0.3*w[0]
		return y