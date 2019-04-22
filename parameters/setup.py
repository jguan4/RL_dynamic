####################################################################
# This file defines the hyperarameters, environment and            #
# architecture used. To change the hyperparameters, only alter     #
# the setup_dict.                                                  #
####################################################################


# fixed_1track_seed defines the seed used to create the fixed one
# track and fixed_3track_seed defines the three seeds used to create
# the fixed three tracks environment
# fixed_1track_seed = [108]
# fixed_3track_seed = [104, 106, 108]

import sys
import os
import numpy as np
path = os.path.realpath(__file__)
sys.path.append(path)

# import necessary classes
from .explore_rates import *
from .learning_rates import * 
from .architectures import *

####################################################################
# Hyperparameters:												   #
# 	architecture requires a string that defines the neural network #
# 		architecture to train on, refer to architectures.py for    #
#		more information.										   #
# 	learning_rate requires a string that defines the learning rate,#
# 		refer to learning_rate.py for more information.			   #
# 	explore_rate requires a string that defines the explore rate   #
#		decay, refer to explore_rates.py for more information.     #
# 	target_update_frequency requires an integer that defines the   #
# 		number of frames between each target Q update;			   #
# 	batch_size requires an integer that defines the size of the    #
#		mini-batch;												   #
# 	memory_capacity requires an integer that defines the capacity  #
#		for replay memory; 										   #
# 	num_episodes requires an integer that defines the number of    #
# 		episodes the algorithm will train on before quitting;      #
# 	learning_rate_drop_frame_limit requires an integer that 	   #
# 		defines the number of frames the exploration rate decays   #
# 		over.													   #
####################################################################
# Environment:													   #
# 	seed defines the seed used for the environment, availvable     #
# 		options include:										   #
#		fixed_1track_seed (fixed one track environment),           #
# 		fixed_3track_seed (fixed three track environment) and      #
# 		None (random tracks environment) 						   #
# 	detect_edges requires a boolean that defines if image 		   #
# 		preprocessing procedure includes edge detection. 		   #
# 	detect_grass requires a boolean that defines if image 		   #
# 		preprocessing procedure includes grass detection. 		   #
# 	flip requires a boolean that defines if images will be flipped #
# 		horizontally during training to include more right curves. #
# 	type requires a string that defines the length of the tracks,  #
# 		available options include ShortTrack (50 tiles track) and  #
# 		None (full track)										   #
####################################################################
setup_dict = {
	'agent': {
		'architecture': Basic_Architecture, 
		'learning_rate': Atari_Learning_Rate,
		'explore_rate': No_Explore,
		'target_update_frequency': 1000,
		'batch_size': 128, 
		'memory_capacity': 1000000, 
		'num_episodes': 10000,
		'learning_rate_drop_frame_limit': 50000
	},

	'acrobot': {
	},

	'pendulum': {
	},

	'henon': {
		'action_range': [+1.0, 0., -1.0],#[+5.0, +2.0, +1.0, 0., -1.0, -2.0, -5.0],
		'delay': False,
		'hs': 0.025
	},

	'lorenz': {
		'action_range': [+2.0, +1.0, 0., -1.0, -2.0],
		'delay': True,
		'hs': 0.025
	},

	'henon_net':{
		'num_act': 17,
		'max_mag': 0.8,
		'act_type': 'line',
		# 'angle': np.pi/4,
		# 'act_dim': [0,1,2],
		'delay': False,
		'num_n': 3,
		'obs': [0,1,2,3,4,5]
	}
}