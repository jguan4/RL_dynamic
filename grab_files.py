import os
import shutil

des_path = '/home/fred/Dropbox/Neural Network/RL/Traj/Henon_Net/'
sour_path = '/home/fred/Documents/JJ_Folder/RL_dynamic/models/0318runs/Henon_Network/'
s_folders = ['Whole_P1_5by2048_3n/']
d_folders = ['Whole_P1_5by2048_3n/']
temp = 'ao_100u_5x64_disp5_batch102u4/'
items = ['x_bar.csv','temp_traj3810.csv']#,'x_bar.csv']
n = len(s_folders)

for ind in range(n):
	temp_s_f = sour_path+s_folders[ind]
	temp_d_f = des_path+d_folders[ind]
	if not os.path.exists(temp_d_f):
		os.makedirs(temp_d_f)
	for item in items:
		shutil.copy( temp_s_f + item, temp_d_f)