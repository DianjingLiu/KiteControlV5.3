import Tracker_new as Tracker
import datetime
import time
from time import sleep

# This is an example for using Interface
if __name__ == "__main__":
    Tracker.READ_FROM_FILE=True
    Tracker.RECORD_LOG = True
    Tracker.RECORD_VIDEO = True
    Tracker.DISPLAY = True
    Tracker.IMAGE_PATH = "G:\\Dropbox\\20180603\\*.bmp"
    data_idx = Tracker.IMAGE_PATH.split('\\')[-2]
    Tracker.LOG_PATH = ".\\log\\"+data_idx + ".npy"
    Tracker.VIDEO_PATH = ".\\log\\"+data_idx + ".mp4"
    trackerthread = Tracker.TrackerThread()

    trackerthread.init_tracker()
    trackerthread.setDaemon(True)
    trackerthread.start()
    print('TrackerThread is starting ...')

    while trackerthread.video_not_end():
        currenttime=datetime.datetime.fromtimestamp(time.time())
        print(currenttime, Tracker.read())
        sleep(0.04)

    trackerthread.stop()
    trackerthread.join()
    print('Tracker thread end.')



