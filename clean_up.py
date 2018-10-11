import os, time, sys

path = os.path.dirname(os.path.realpath(__file__))

best_chkp_time_p = time.mktime(time.strptime("Tue Oct 2 17:38 2018","%a %b %d %H:%M %Y"))
best_chkp_time_a = time.mktime(time.strptime("Tue Oct 2 17:39 2018","%a %b %d %H:%M %Y"))

files = os.listdir(path)
paths = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and \
'data.chkp' in i]
last_modified = max(paths, key=os.path.getctime)

for f in os.listdir(paths):
	if os.stat(os.path.join(path,f)).st_mtime < best_chkp_time_a and \
	os.stat(os.path.join(path,f)).st_mtime > best_chkp_time_p:
		os.rename(f,"best_chkp")
	else if os.stat(os.path.join(path,f)) == last_modified:
		os.rename(f,"last_chkp")	
	else:
		if os.path.isfile(f):
			os.remove(os.path.join(path, f))