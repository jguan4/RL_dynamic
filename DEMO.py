import subprocess
import numpy as np
import os
import utils
import re
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

factors = [1, 2, 4, 6, 8, 10]
normalized_1 = 97*np.square(np.pi)
normalizes = [normalized_1*4, normalized_1*2, normalized_1, normalized_1*0.5, normalized_1*0.25]

train = 1
# for factor in factors:
# 	for normalize in normalizes:
factor = factors[0]
normalize = normalizes[2]
model_name = "1106runs/point5torque".format(factor, normalize/normalized_1)
model_path = DIR_PATH+"/models/"+model_name
# if os.path.isdir(model_path):
# 	continue
# else:

best_checkpoint_name = 'None'
if not train:
	files = os.listdir(model_path)
	paths = [i for i in files if os.path.isfile(os.path.join(model_path,i)) and re.match('best.data.chkp', i)]
	best_checkpoint_name = os.path.join(model_path,os.path.splitext(paths[1])[0])


subprocess.run(['python3', 'main.py', str(factor), str(normalize), model_name, best_checkpoint_name])