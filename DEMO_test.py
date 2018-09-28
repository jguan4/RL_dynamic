import subprocess
import numpy as np

factors = [1, 2, 4, 6, 8, 10]
normalized_1 = 97*np.square(np.pi)
normalizes = [normalized_1*4, normalized_1*2, normalized_1, normalized_1*0.5, normalized_1*0.25]

factor = factors[0]
normalize = normalizes[2]
checkpoint = 2761
model_name = "Acrobot_reward_factor_{0}/normalize_{1}".format(factor, normalize/normalized_1)
subprocess.run(['python3', 'tester.py', str(factor), str(normalize), model_name, str(checkpoint)])