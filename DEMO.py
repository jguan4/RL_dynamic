import subprocess
import numpy as np
import os
import utils
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

factors = [1, 2, 4, 6, 8, 10]
normalized_1 = 97*np.square(np.pi)
normalizes = [normalized_1*4, normalized_1*2, normalized_1, normalized_1*0.5, normalized_1*0.25]

# for factor in factors:
# 	for normalize in normalizes:
factor = factors[0]
normalize = normalizes[2]
model_name = "1008runs/Pendulum/noactioninreward".format(factor, normalize/normalized_1)
model_path = DIR_PATH+"/models/"+model_name
# if os.path.isdir(model_path):
# 	continue
# else:
subprocess.run(['python3', 'main.py', str(factor), str(normalize), model_name])
