#import numpy as np
from Kite_Tracking_angle_detection_16fps.interface import *
#from Kite_Tracking_angle_detection_16fps.record import my_recorder
from pdb import set_trace
import datetime

class my_recorder:
	def __init__(self):
		# [timestamp, done, x, y, phi]
		return

	def add(self, filename, location, ok, phi):
		# filename: 
		time = filename.split('-')
		time = list(map(int, time[0:-1])) # python 3
		# time = map(int, time[0:-1]) # python 2
		timestamp = datetime.datetime(time[0],time[1],time[2],time[3],time[4],time[5],time[6]*1000).timestamp()
		
		
		if phi == None:
			ok = False
			phi = 0

		if ok:
			x = location[0]
			y = location[1]
		else:
			x = 0
			y = 0
		log = np.array([timestamp, 1-ok, x, y, phi])
		#print(log)
		#print(log.shape)
		
		try:
			self.tot_log = np.append(self.tot_log, log[np.newaxis,:], axis=0)
		except AttributeError:
			self.tot_log = log[np.newaxis, :]
		

	def save(self, filename):
		#np.savez(filename, log=self.tot_log)
		#np.savetxt('record20180118.txt', self.tot_log, delimiter=',')
		log = self.tot_log
		np.save(filename, log)



# This is an example for using Interface
# To avoid opening opencv window and verbose information, 
# please set the variables:
#           WRITE_TMP_RESULT = True
#           DEBUG_MODE = False
# 
WRITE_TMP_RESULT = True
DEBUG_MODE = False
# IMAGE_PATH = "./test/*.bmp"
IMAGE_PATH = "F:\\GrapV1.13\\Dest\\*.bmp"



if __name__ == "__main__":
	# Read video
	files = glob.glob(IMAGE_PATH)
	assert len(files) > 0

	_, path_and_file = os.path.splitdrive(files[0])
	path, file = os.path.split(path_and_file)

	video = Video(files, FILE_FORMAT, START_FRAME)
	frame_num = video.getFrameNumber()
	ok, frame = video.read()
	if not ok:
		print( 'Cannot read video file')
		sys.exit()

	tracker = Interface()
	tracker.init_tracker(frame)
	recorder = my_recorder()

	while True:
		# Read one frame
		ok, frame = video.read()
		if not ok:
			break

		# Obtain results
		ok, bbox, angle, center_loc = tracker.update(frame, verbose=False)
		filename = video.getFrameName()
		recorder.add(filename, center_loc, ok, angle)
		if ok:
			print( "image {} / {}  |  anlge: {:3d}  |  center: {:4d} {:4d}".format(
																					video.getFrameIdx(), frame_num,
																					#int(bbox[0]), int(bbox[1]), 
																					#int(bbox[2]), int(bbox[3]),
																					int(angle),
																					center_loc[0], center_loc[1]) 
			)
			"""
			drawBox(frame, bbox)
			drawAnlge(frame, angle, bbox)
			drawPoint(frame, center_loc)
			frame_resize = cv2.resize(frame, (512, 512))
			cv2.imshow("frame", frame_resize)
			cv2.waitKey(1)
			"""
		else:
			print( "   ->Tracking failed!!!")
	recorder.save('.\\log\\record_kite'+ path.split('\\')[-1] +'.npy')
