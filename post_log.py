
from pdb import set_trace
import glob
import os
import numpy as np
import cv2
import imageio
import time

AI_MODEL = -1
#DIR_PATH = "C:\\Users\\ldj\\Documents\\MachineLearning_NN\\kite\\data\\20180603"
DIR_PATH = "C:\\Users\\ldj\\Documents\\MachineLearning_NN\\kite\\ControlV5.3\\log\\20180617"
VIDEO_PATH = os.path.join(DIR_PATH, '20180518_AI{:d}.mp4'.format(AI_MODEL))
#VIDEO_PATH = os.path.join(DIR_PATH, 'speedup20180329.mp4')
if AI_MODEL == -1:
	MODE = 1 # Manual control
	LOG_PATH = os.path.join(DIR_PATH, "*.txt")
	SAVE_PATH = os.path.join(DIR_PATH, 'log_notrack_20180617.npy')

else:
	MODE = 2 # Auto control
	LOG_PATH = os.path.join(DIR_PATH, "*_AI{:d}.txt".format(AI_MODEL) )
	SAVE_PATH = os.path.join(DIR_PATH, 'record20180416_AI{:d}.npy'.format(AI_MODEL) )

def process_log(log_path, mode, save_path=None):
	def str2float(v):
		return float(v.lower() in (b"yes", b"true", b"t", b"1"))
	# Read video
	files = glob.glob(log_path)
	assert len(files) > 0
	log = []
	for file in files:
		#a=np.loadtxt(file, delimiter=', ', usecols=(0,1,6,7,8,9,10,11,12))
		if len(np.loadtxt(file,delimiter=', ',usecols=1))>0:
			a=np.loadtxt(file, delimiter=', ', converters={2:str2float})
			log.append(a)
	log=np.concatenate(tuple(log), axis=0).astype(float) # [timestamp, MODE, OK, x, y, phi, l1, l2, v1, v2, T1, T2, J]
	# select correct MODE
	log = log[log[:,1]==mode]
	log = np.delete(log, 1, 1) # [timestamp, OK, x, y, phi, l1, l2, v1, v2, T1, T2, J]
	
	# Process log
	log = np.nan_to_num(log)
	log[:,1] = 1 - log[:,1] # convert 'OK' to 'done'
	log[:,2:4] = 0.001*(log[:,2:4] - 700) # normalize (x, y)
	log[:,4] = log[:,4] * np.pi / 180.0 # normalize phi
	if save_path is not None:
		print("Save log to {}".format(save_path))
		np.save(save_path, log)
	print('DONE/ALL: {:d}/{:d}'.format( np.sum(log[:,1]==1), len(log) ))
	return log

def replay(log, video_path, BG_PATH='bg.bmp'):
	def drawAnlge(frame, angle, center, length=25):
		"""
		Draw angle axis.
	
		Params:
			frame: current image
			angle: float value
			center: kite center (x, y)
			length: length of axis
		Return:
		    Painted image
		"""
		radian = angle / 180 * np.pi
		vertice = (int(center[0] + np.cos(radian)*length), int(center[1] + np.sin(radian)*length))
		frame = cv2.line(frame, center, vertice, (0, 0, 255), 5)
		return frame
	def drawPoint(frame, point, color=(255, 0, 255), radius=10):
		return cv2.circle(frame, tuple(point), radius, color, -1)

	width = 512
	height = 512
	img_bg = cv2.imread(BG_PATH)
	#fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
	#out = cv2.VideoWriter(video_path, fourcc, 25.0, (width, height))
	video_writer = imageio.get_writer(video_path, fps=25)
	last_done = 1
	x, y, phi = 0,0,0
	for record in log:
		if record[1]==0:
			last_done = 0
			x = int(record[2]*1000+700)
			y = int(record[3]*1000+700)
			phi = record[4]*180/np.pi
			frame = drawAnlge(np.copy(img_bg), phi, (x,y))
			frame = drawPoint(frame, (x,y))
			frame = cv2.resize(frame, (width, height))
			video_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		else:
			if last_done == 0:
				last_done = 1
				frame = drawAnlge(np.copy(img_bg), phi, (x,y))
				frame = drawPoint(frame, (x,y),color=(255, 255, 255))
				frame = cv2.resize(frame, (width, height))
				#frame = np.zeros((width,height,3))
				video_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
				video_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

	video_writer.close()

	#out.release()

if __name__ == "__main__":
	log = process_log(LOG_PATH, MODE, SAVE_PATH)
	#log = process_log(LOG_PATH, MODE)
	#replay(log,VIDEO_PATH)
	
	#log = np.load("C:\\Users\\ldj\\Documents\\MachineLearning_NN\\kite\\data\\records\\record20180407.npy")
	#replay(log,"C:\\Users\\ldj\\Documents\\MachineLearning_NN\\kite\\data\\records\\20180407.mp4")
	