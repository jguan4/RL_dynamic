import sys
sys.dont_write_bytecode = True

import sys
import tensorflow as tf
import agent
import environment as env
import parameters.setup as setup

#####################################  Usage  ##########################################################
# 1) A command line argument specifying the name of the folder we want to log in must
#    be specified when this file is run, like so: "python main.py name_of_folder".
# 2) The parameters for DQN_Agent and CarRacing are defined in the setup_dict object 
#    in parameters/setup.py.
########################################################################################################

environment = env.Acrobot(**setup.setup_dict['acrobot'], factor = sys.argv[1], normalize = sys.argv[2])
control = agent.DQN_Agent(environment=environment, model_name=sys.argv[3], **setup.setup_dict['agent'])

#####################################  Traning a model  ################################################
control.train()

#####################################  Testing a model  ################################################
##### 
# control.load("/home/fred/Documents/JJ_Folder/RL_dynamic/models/acrobot_reward_pendulum_normalize_correcte/data.chkp-2791")
# control.test(5, True)
