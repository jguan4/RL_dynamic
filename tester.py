import sys
sys.dont_write_bytecode = True
import os
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
import tensorflow as tf
import agent
import environment as env
import parameters.setup as setup

environment = env.Acrobot(**setup.setup_dict['acrobot'], factor = int(sys.argv[1]), normalize = float(sys.argv[2]))
control = agent.DQN_Agent(environment=environment, model_name=sys.argv[3], **setup.setup_dict['agent'])

#####################################  Testing a model  ################################################
##### 
print(path)
control.load(path+"/models/"+sys.argv[3]+"/data.chkp-"+sys.argv[4])
control.test(5, True, pause = True)