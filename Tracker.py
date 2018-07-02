import cv2
import sys
import os
import imageio
import glob
import copy
import time
import numpy as np
import threading
import signal

import mmap
import contextlib
import struct
import datetime
from threading import *
"""
from MLP import MLP_Detection_MP
from video import Video
from matched_filters import MatchedFilter
from utils import * 
from config import *
from bs import BS
"""
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TRACKER_PATH = os.path.join(DIR_PATH, "Kite_Tracking_speedup")
sys.path.insert(0,TRACKER_PATH)
from Kite_Tracking_speedup.interface import *

# Params for reading image from memory
FULL_MAP_NAME="Local\FramGrapMap"

CAP_PIC_WIDTH=1504
CAP_PIC_HEIGHT=1496

KITE_CROP_WIDTH=36
KITE_CROP_HEIGHT=36

VIEW1_OFFSET=0
VIEW1_SIZE=1024

VIEW2_OFFSET=VIEW1_OFFSET+VIEW1_SIZE
VIEW2_SIZE=KITE_CROP_WIDTH * KITE_CROP_HEIGHT * 4

VIEW3_OFFSET=VIEW2_OFFSET+VIEW2_SIZE
VIEW3_SIZE=CAP_PIC_WIDTH* CAP_PIC_HEIGHT * 4

FULL_MAP_SIZE=VIEW3_OFFSET+VIEW3_SIZE

# Number of frames for tracker initialization
INIT_FRAMES_NUM=10

ProgramRunSpeed = 0.04  # 刷新频率，间隔0.1秒

READ_FROM_FILE = False
DISPLAY=False
RECORD_LOG = False
LOG_PATH = "./log/track.npy"
RECORD_VIDEO = False
VIDEO_PATH = "./log/track.mp4"

RUNTIME = 0
ok = False
x = 0
y = 0
phi = 0
def read():
    return [ok, x, y, phi]




class TrackerThread(Thread):
    def run(self):
        global ok, x, y, phi, RUNTIME
        timestamp = time.time()
        self.ifdo = True
        self.init_tracker()
        # Record variables
        video_writer = imageio.get_writer(VIDEO_PATH, fps=RECORD_FPS)
        log = []
        while self.ifdo:
            RUNTIME = 1000*(time.time()-timestamp)
            timestamp = time.time()
            frame = self.get_img(READ_FROM_FILE)
            #frame_original = frame.copy()
    
            # Obtain results
            #timestamp = time.time()
            ok, bbox, angle, center_loc, fps = self.tracker.update(frame, verbose=False)
            #print("Run time:{:2f}".format(1000*(time.time()-timestamp)))
            if ok:
                x, y, phi = center_loc[0], center_loc[1], angle
            """
            # Print result on terminal
            cnn_pred = self.tracker.cnn_pred
            print("Frame: {:5d} | bbox: {:4d} {:4d} {:3d} {:3d}  | fps: {:3d}  |  anlge: {:3d}  |  CNN predict: {}".format(
                                                    self.video.getFrameIdx(), 
                                                    int(bbox[0]), int(bbox[1]), 
                                                    int(bbox[2]), int(bbox[3]),
                                                    int(fps),
                                                    int(angle),
                                                    cnn_pred)) 
            """
            # Make timestamp and save log
            if RECORD_LOG:
                filename = self.video.getFrameName()
                rec_time = filename.split('-')
                rec_time = list(map(int, rec_time[0:-1])) # python 3
                # rec_time = map(int, rec_time[0:-1]) # python 2
                timestamp = datetime.datetime(rec_time[0],rec_time[1],rec_time[2],rec_time[3],rec_time[4],rec_time[5],rec_time[6]*1000).timestamp()
                log.append([timestamp, 1-ok, x,y,phi])
            # Plot result image, display or save as video
            if DISPLAY or RECORD_VIDEO:
                frame_resize = self.plot_result(frame, ok, bbox, center_loc, angle, fps)
            if RECORD_VIDEO:
                video_writer.append_data(swapChannels(frame_resize))
            if DISPLAY:
                cv2.imshow("frame", frame_resize)
                k = cv2.waitKey(1)
                if k == 27 : self.stop()# Press 'ESC' to quit

        if RECORD_VIDEO:
            print("Save image to {}".format(VIDEO_PATH))
            video_writer.close()
        if RECORD_LOG:
            print("Save log to {}".format(LOG_PATH))
            np.save(LOG_PATH, log)

    def stop(self):
        self.ifdo = False

    def init_tracker(self, init_frames=None):
        # Define and initialize tracker
        print('Initializing tracker...')
        self.tracker = Interface()
        if init_frames is None:
            init_frames = []
            for _ in range(INIT_FRAMES_NUM):
                st_time = time.time()
                init_frames.append( self.get_img(from_file=READ_FROM_FILE))
                slp_time = ProgramRunSpeed - (time.time()-st_time) 
                if slp_time>0:
                    time.sleep( slp_time)
        if len(init_frames)==0:
            print('Cannot find images!')
        self.tracker.init_tracker(init_frames)
        self.tracker.update(init_frames[-1])
        print('Tracker initialized.')   
    def get_img(self, from_file=READ_FROM_FILE):
        # For DEBUG: read image from file
        if from_file == True:
            return self.get_img_from_file(IMAGE_PATH)
        # Read camera image from memory
        # Run time ~10.5ms
        with contextlib.closing(mmap.mmap(-1, FULL_MAP_SIZE, tagname=FULL_MAP_NAME, access=mmap.ACCESS_READ)) as SharedMemry:
            # Read kite info
            CropPkg = SharedMemry.read(20+36+24)
            [SerialNum, 
            wYear, wMonth, wDayOfWeek, wDay,wHour, wMinute,wSecond,wMilliseconds,
            bFound,Xcoordinate,Ycoordinate,KiteWidth,KiteHeight,KiteRotateAngle,
            Radius,Angle,dConArea,
            KiteImageWidth,KiteImageHeight,KiteImageChannels,OriginalImageWidth,OriginalImageHeight,OriginalImageChannels]= struct.unpack('IHHHHHHHHiffffffffiiiiii', CropPkg)
            # Read original image
            SharedMemry.seek(VIEW3_OFFSET)
            KiteImage=SharedMemry.read(OriginalImageWidth*OriginalImageHeight*OriginalImageChannels)    # Run time ~4ms
            nparr = np.fromstring(KiteImage, np.uint8).reshape((OriginalImageHeight, OriginalImageWidth, OriginalImageChannels )) # Run time ~4ms
            #nparr = cv2.cvtColor(nparr, cv2.COLOR_BGR2RGB) # Run time ~2ms
        return nparr
    def get_img_from_file(self, img_path=IMAGE_PATH):
        if not hasattr(self, 'video'):
            files = glob.glob(img_path)
            if len(files) == 0:
                print('Can not find image in path '+ img_path + '\nTracking thread will stop ...')
            assert len(files) > 0
            _, path_and_file = os.path.splitdrive(files[0])
            path, file = os.path.split(path_and_file)
            self.video = Video(files, FILE_FORMAT, START_FRAME)
        ok, frame = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        else:
            return frame
    def video_not_end(self):
        if not hasattr(self, 'video'):
            return True
        else:
            return self.video.getFrameNumber() - 1 > self.video.getFrameIdx()
    def plot_result(self, frame, ok, bbox, center_loc, angle, fps):
        if ok:
            drawBox(frame, bbox)
            drawAnlge(frame, angle, center_loc)
            drawPoint(frame, center_loc)
            cv2.putText(frame, "Angle : " + str(int(angle)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 
                        (0, 255, 0), 5);
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 
                        (0, 255, 0), 5);    
        else:
            print("Fail on tracking!!!")
            cv2.putText(frame, "Fail!", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 
                        (0, 0, 255), 5);
        frame_resize = cv2.resize(frame, (512, 512))
        return frame_resize

# This is an example for using Interface
if __name__ == "__main__":
    READ_FROM_FILE=True
    trackerthread = TrackerThread()
    trackerthread.run()





