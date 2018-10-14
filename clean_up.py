import os, time, sys, re,utils

path = "/home/fred/Documents/JJ_Folder/RL_dynamic/models/1001runs/Acrobot_reward_factor_1_straightstart/nostack/neural512"

best_chkp_time_p = time.mktime(time.strptime("Wed Oct 3 10:11 2018","%a %b %d %H:%M %Y"))
best_chkp_time_a = time.mktime(time.strptime("Wed Oct 3 10:12 2018","%a %b %d %H:%M %Y"))

files = os.listdir(path)
paths = [os.path.join(path,i) for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and re.match('data.chkp', i)]
last_modified = max(paths, key=os.path.getctime)
last_modified_name = os.path.splitext(os.path.basename(last_modified))[0]

for f in paths:
	if os.stat(f).st_mtime < best_chkp_time_a and \
	os.stat(f).st_mtime > best_chkp_time_p:
		os.rename(f,os.path.join(path,"best."+os.path.basename(f)))
	elif re.match(last_modified_name,os.path.basename(f)):
		os.rename(f,os.path.join(path,"last."+os.path.basename(f)))
	else:
		if os.path.isfile(f):
			os.remove(f)