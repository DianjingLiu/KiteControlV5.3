import numpy as np
import os
#import matplotlib.pyplot as plt
#DIR_PATH = "C:\\Users\\ldj\\Documents\\MachineLearning_NN\\kite\\ControlV5.3\\log"
DIR_PATH = "C:\\Users\\ldj\\Documents\\MachineLearning_NN\\kite\\data\\20180617"
data_idx = DIR_PATH.split("\\")[-1]#os.path.split(DIR_PATH,'\\')
print(data_idx)
LOG_PATH = os.path.join(DIR_PATH, "log_notrack_{}.npy".format(data_idx))
TRACK_PATH = os.path.join(DIR_PATH, "track_{}.npy".format(data_idx))
SAVE_PATH = os.path.join(DIR_PATH, "record{}.npy".format(data_idx))

def match_log(log, timestamp, inc_time = False):
	"""
	Find closest log of the given timestamp

	Params:
		log: [timestamp, ...]
		inc_time: if True, include time data in select_log
				  if False, return select_log[:, 1:]
	"""
	
	log_time = log[:, 0]
	n_case = len(timestamp)
	select_log = np.zeros((n_case, log.shape[1]))
	wrong_idx = []
	for i in range(n_case):
		diff = np.absolute(timestamp[i] - log_time)
		idx = np.argmin(diff)
		select_log[i] = log[idx]
		if diff[idx]>0.02:
			wrong_idx.append(i)
			#print('Log match failure: {0:.0f}:{1:.0f}:{2}'.format(timestamp[i]))
	if inc_time == True:
		return select_log, wrong_idx
	else:
		return select_log[:,1:], wrong_idx

if __name__ == '__main__':
	log = np.load(LOG_PATH)
	track = np.load(TRACK_PATH)
	track_timestamp = track[:,0]

	track[:,2:4] = 0.001*(track[:,2:4] - 700)
	track[:,4] = track[:,4] * np.pi / 180.0

	record, wrong_idx = match_log(log, track_timestamp, True)
	record[:, :5] = track
	record = np.delete(record, wrong_idx, axis=0)
	print("{} records. {} track succeed. {} failed to match.".format(len(track_timestamp), np.sum(record[:,1]==0), len(wrong_idx)))
	print("Save record to {}".format(SAVE_PATH))
	np.save(SAVE_PATH, record) # [timestamp, done, x, y, phi, l1, l2, v1, v2, T1, T2, J]
