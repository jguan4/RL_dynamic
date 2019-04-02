import os
import shutil

des_path = '/home/fred/Dropbox/Neural Network/RL/Traj/Henon/'
sour_path = '/home/fred/Documents/JJ_Folder/RL_dynamic/models/0318runs/Henon/'
s_folders = ['Delay_P1_25by2048/','Delay_P1_10by2048/','Delay_P1_5by2048/','Delay_P1_1by2048/']
d_folders = ['Delay_P1_25by2048/','Delay_P1_10by2048/','Delay_P1_5by2048/','Delay_P1_1by2048/']
temp = 'ao_100u_5x64_disp5_batch1024/'
items = ['frame_eps.csv','training_scores.csv']
n = len(s_folders)

for ind in range(n):
	temp_s_f = sour_path+s_folders[ind]
	temp_d_f = des_path+d_folders[ind]
	if not os.path.exists(temp_d_f):
		os.makedirs(temp_d_f)
	for item in items:
		shutil.copy( temp_s_f + item, temp_d_f)