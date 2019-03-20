import os
import shutil

des_path = '/home/fred/Dropbox/Neural Network/RL/Traj/Henon/'
sour_path = '/home/fred/Documents/JJ_Folder/RL_dynamic/models/0303runs/Henon_FE/'
s_folders = ['Whole_P1/','Whole_P2/','Whole_P4/']
d_folders = ['Whole_P1_FE/','Whole_P2_FE/','Whole_P4_FE/']
items = ['frame_eps.csv','period_points.csv','training_scores.csv','traj.csv']
n = len(s_folders)

for ind in range(n):
	temp_s_f = sour_path+s_folders[ind]
	temp_d_f = des_path+d_folders[ind]
	for item in items:
		shutil.copy( temp_s_f + item, temp_d_f)