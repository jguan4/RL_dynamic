import os
import utils

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
period = 1
model_name = "0205runs/Henon_paper_NE/Delay_P{0}/ao_100u_5x64_disp5_batch1024_rewardfordiffer_ver10".format(period)
model_path = DIR_PATH+ '/models/' + model_name 
plot_file = model_path+"/test_traj.csv"
utils.plot_func(plot_file)