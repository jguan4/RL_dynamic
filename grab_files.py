import os
import shutil

des_path = '/home/fred/Dropbox/Neural Network/RL/Traj/Henon/'
sour_path = '/home/fred/Documents/JJ_Folder/RL_dynamic/models/0221runs/Henon_paper_FE/'
s_folders = ['Delay_P4/']
d_folders = ['Delay_P4_FE/']
temp = 'ao_100u_5x64_disp5_batch1024/'
items = ['frame_eps.csv','period_points.csv','training_scores.csv','traj.csv']
n = len(s_folders)

for ind in range(n):
	temp_s_f = sour_path+s_folders[ind]
	temp_d_f = des_path+d_folders[ind]
	for item in items:
		shutil.copy( temp_s_f + temp + item, temp_d_f)