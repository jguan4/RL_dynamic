import numpy as np
from numpy import linalg as LA

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
		obs_arr_t = np.zeros(self.num_n*self.dim)
		obs_arr_t[[self.obs]] = 1
		self.obs_arr = np.tile(obs_arr_t,(self.num_n*self.dim,1))

	def create_net_action(self, action_range, act_dim):
		act_ref = np.empty((0,self.num_n*self.dim),float)
		for i in act_dim:
			basis = np.zeros(self.num_n*self.dim)
			basis[i] = 1
			act_set = np.outer(action_range,basis)
			act_ref = np.append(act_ref,act_set,axis=0)
		act_ref = np.append(act_ref,[np.zeros(self.num_n*self.dim)],axis=0)
		return act_ref

	def create_action_range(self,num_act,max_mag,act_type):
		n = np.floor(num_act/2)
		if act_type=='dense':
			mul = max_mag*(n+1)/n
			h = np.subtract(np.divide(1,np.arange(n+1,0,-1)),1/(n+1))
			hh = np.multiply(mul,h)
			hh_n = -np.flip(hh,axis=0)
			action_range = np.append(hh_n[0:int(n)],hh)
		elif act_type== 'line':
			incre = max_mag/n
			h = np.arange(-n, n+1)
			action_range = np.multiply(incre,h)
		return action_range

	def create_line_action(self, action_range):
		temp_b = np.append(np.ones(self.num_n),np.zeros(self.num_n))
		basis = temp_b/LA.norm(temp_b)
		act_ref = np.outer(action_range,basis)
		return act_ref