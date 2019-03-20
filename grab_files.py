import os
import shutil

des_path = '/home/fred/Dropbox/Neural Network/RL/Traj/Henon_Net/'
sour_path = '/home/fred/Documents/JJ_Folder/RL_dynamic/models/0318runs/Henon_Network/'
s_folders = ['Whole_P1_2n/','Whole_P1_2n_obs1/','Delay_P1_2n/','Delay_P2_2n/']
d_folders = ['Whole_P1_2n/','Whole_P1_2n_obs1/','Delay_P1_2n/','Delay_P2_2n/']
temp = 'ao_100u_5x64_disp5_batch1024/'
items = ['x_bar.csv','temp_traj1150.csv']
n = len(s_folders)

for ind in range(n):
	temp_s_f = sour_path+s_folders[ind]
	temp_d_f = des_path+d_folders[ind]
	if not os.path.exists(temp_d_f):
		os.makedirs(temp_d_f)
	for item in items:
		shutil.copy( temp_s_f + temp + item, temp_d_f)