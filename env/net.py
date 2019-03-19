import numpy as np

def create_full_adj(num_n):
	adj = np.ones((num_n,num_n))
	inds = range(num_n)
	adj[inds,inds] = 0
	return adj

def create_obs_arr(num_n,dim,obs):
	arr = np.zeros(num_n*dim)
	arr[[obs]] = 1
	return arr