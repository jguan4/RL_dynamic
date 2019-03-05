import numpy as np
from numpy import linalg as LA
import utils

class Find_Period:
	def __init__(self, period):
		self.tol = 1e-10
		self.period = period
		init_state =  [0.83, 0.8] #+ np.random.normal(0, 0.1, 2)
		self.xs = [init_state]
		ns_p = self.henon(self.xs[-1])
		self.xs = np.append(self.xs,[ns_p[-1]],axis=0)

	def broyden(self,k):
		b = np.identity(2)
		for i in range(k):
			x_0 = self.xs[-1]
			x = x_0-np.transpose(np.matmul(b,self.f_func(x_0)))
			self.xs = np.append(self.xs,[x],axis=0)
			de = np.transpose(x-x_0)
			delta = self.f_func(x)-self.f_func(x_0)
			s1 = np.matmul(b,delta)
			s2 = de-s1
			s3 = np.matmul(s2,np.transpose(de))
			s4 = np.matmul(s3,b)
			s5 = np.matmul(np.transpose(de),b)
			s6 = np.matmul(s5,delta)
			b=b+s4/s6
		whole_traj = self.henon(self.xs[-1])
		return whole_traj

	def run_secant(self):
		err = 1
		while err>self.tol:
			x_n = self.secant()
			self.xs = np.append(self.xs,[x_n],axis=0)
			diff = self.xs[-1]-self.xs[-2]
			err = LA.norm(diff,2)
			print("New State {0} \t Error: {1}".format(x_n,err))
		whole_traj = self.henon(self.xs[-1])
		return whole_traj

	def secant(self):
		x_2=self.xs[-2].copy()
		x_1=self.xs[-1].copy()
		f_x_1 = self.f_func(x_1)
		f_x_2 = self.f_func(x_2)
		x_n = x_1-f_x_1*(x_1-x_2)/(f_x_1-f_x_2)
		return x_n

	def f_func(self, w):
		w_n = self.henon(w)
		diff = w-w_n[-1]
		# diff_I = np.argmax(np.absolute(diff))
		# return diff[diff_I]
		return np.transpose(diff)

	def henon(self,w):
		y = np.zeros((self.period,2))
		# y[0] = -1.4*np.square(w[0])+w[1]+1
		# y[1] = 0.3*w[0]
		for i in range(self.period):
			# y[0] = 2*np.cos(w[0])+0.4*w[1]
			# y[1] = w[0]
			y[i,0] = 1.29+0.3*w[1]-w[0]**2
			y[i,1] = w[0]
			w = y[i,:].copy()
		y = np.array(y)
		return y