import sys
sys.dont_write_bytecode = True
import os
path = os.path.realpath(__file__)
sys.path.append(path)
import tensorflow as tf
import agent
import environment as env
import numpy as np
import parameters.setup as setup
import ast


#####################################  Usage  ##########################################################
# 1) A command line argument specifying the name of the folder we want to log in must
#    be specified when this file is run, like so: "python main.py name_of_folder".
# 2) The parameters for DQN_Agent and CarRacing are defined in the setup_dict object 
#    in parameters/setup.py.
########################################################################################################

environment = env.Henon_Map(**setup.setup_dict['henon'], direction = ast.literal_eval(sys.argv[1]), 
	period = float(sys.argv[2]), mag = float(sys.argv[3]))
control = agent.DQN_Agent(environment=environment, model_name=sys.argv[4], **setup.setup_dict['agent'])

#####################################  Traning a model  ################################################
if sys.argv[5]=='None':
	control.train()
else:
##################################  Testing a checkpoint ###############################################
	control.load(sys.argv[5])
	control.test(5, True, pause = True)