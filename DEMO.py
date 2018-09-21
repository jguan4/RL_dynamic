import subprocess
import numpy as np

factors = [1, 2, 4, 6, 8, 10]
normalized_1 = 97*np.square(np.pi)
normalizes = [normalized_1, normalized_1*0.5, normalized_1*0.25, normalized_1*0.1]

for factor in factors:
	for normalize in normalizes:
		model_name = "Acrobot_reward_factor_{0}/normalize_{1}".format(factor, normalize)
		subprocess.run(['python3', 'main.py', str(factor), str(normalize), model_name])
