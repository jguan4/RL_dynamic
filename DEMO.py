import subprocess
import numpy as np
import os
import utils
import re
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# factors = [1, 2, 4, 6, 8, 10]
# normalized_1 = 97*np.square(np.pi)
# normalizes = [normalized_1*4, normalized_1*2, normalized_1, normalized_1*0.5, normalized_1*0.25]
# for factor in factors:
# 	for normalize in normalizes:
# factor = factors[0]
# normalize = normalizes[2]

# direction_ang = [0,np.pi/6,np.pi/4,np.pi/3,np.pi/2]
# direction_ind = 0
# direction = [np.cos(direction_ang[direction_ind]), np.sin(direction_ang[direction_ind])]
period = 1
model_name = "0318runs/Henon_Network/Delay_P{0}_2n_9a/ao_100u_5x64_disp5_batch1024".format(period)
model_path = DIR_PATH+"/models/"+model_name
# utils.plot_func(model_path+'/temp_traj390.csv')

train = 1
best_checkpoint_name = 'None'
if not train:
	files = os.listdir(model_path)
	paths = [i for i in files if os.path.isfile(os.path.join(model_path,i)) and re.match('best.data.chkp', i)]
	best_checkpoint_name = os.path.join(model_path,os.path.splitext(paths[1])[0])

subprocess.run(['python3', 'main.py', str(period), model_name, best_checkpoint_name])
