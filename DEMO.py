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

mags = [0.01,0.025]
mags_string = ['1e-2','25e-3']
mag_ind = 1
mag = mags[mag_ind]

direction_ang = [0,np.pi/6,np.pi/4,np.pi/3,np.pi/2]
direction_ind = 0
direction = [np.cos(direction_ang[direction_ind]), np.sin(direction_ang[direction_ind])]

train = 1
model_name = "0103runs/Henon_paper_NE/128_5_layers_64_tracking_average".format(mags_string[mag_ind], direction_ang[direction_ind])
model_path = DIR_PATH+"/models/"+model_name

best_checkpoint_name = 'None'
if not train:
	files = os.listdir(model_path)
	paths = [i for i in files if os.path.isfile(os.path.join(model_path,i)) and re.match('best.data.chkp', i)]
	best_checkpoint_name = os.path.join(model_path,os.path.splitext(paths[1])[0])

factor=direction
normalize=0
subprocess.run(['python3', 'main.py', str(factor), str(normalize), str(mag), model_name, best_checkpoint_name])