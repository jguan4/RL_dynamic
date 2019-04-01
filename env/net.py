import numpy as np

class Net:
	def __init__(self,num_n,dim,obs):
		self.num_n = num_n
		self.dim = dim
		self.obs = obs
		self.obs_num = len(obs)
		self.create_full_adj()
		self.create_obs_arr()

	def create_full_adj(self):
		self.adj = np.ones((self.num_n,self.num_n))
		inds = range(self.num_n)
		self.adj[inds,inds] = 0

	def create_obs_arr(self):
		self.obs_arr = np.zeros(self.num_n*self.dim)
		self.obs_arr[[self.obs]] = 1

	def create_net_action(self, action_range, act_dim):
		act_ref = np.empty((0,self.num_n*self.dim),float)
		for i in act_dim:
			basis = np.zeros(self.num_n*self.dim)
			basis[i] = 1
			act_set = np.outer(action_range,basis)
			act_ref = np.append(act_ref,act_set,axis=0)
		act_ref = np.append(act_ref,[np.zeros(self.num_n*self.dim)],axis=0)
		return act_ref

	def create_line_action(self, angle, action_range):
		basis = [np.cos(angle),np.cos(angle),0,0]
		act_ref = np.outer(action_range,basis)
		return act_ref