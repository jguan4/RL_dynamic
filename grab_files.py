import os
import shutil

des_path = '/home/fred/Dropbox/Neural Network/RL/Traj/Henon/'
sour_path = '/home/fred/Documents/JJ_Folder/RL_dynamic/models/0318runs/Henon/'
s_folders = ['Whole_P1_5by2048_Explore2000/']
d_folders = ['Whole_P1_5by2048_Explore2000/']
temp = 'ao_100u_5x64_disp5_batch1024/'
items = ['frame_eps.csv']#,'x_bar.csv']
n = len(s_folders)

for ind in range(n):
	temp_s_f = sour_path+s_folders[ind]
	temp_d_f = des_path+d_folders[ind]
	if not os.path.exists(temp_d_f):
		os.makedirs(temp_d_f)
	for item in items:
		shutil.copy( temp_s_f + item, temp_d_f)