#  -*- coding:utf-8 -*-  
import serial
import struct
from threading import *
import time
from time import sleep

# 创建serial实例
SensorPort = serial.Serial()
SensorPort.port = 'COM6'
SensorPort.baudrate = 115200
SensorPort.parity = 'N'
SensorPort.bytesize = 8
SensorPort.stopbits = 1
SensorPort.timeout = 0.001

ProgramRunSpeed = 0.1  # 刷新频率，间隔0.1秒

RevolutionsL = 0  # 取值范围:1代表一圈，约28cm
RevolutionsR = 0  # 取值范围:1代表一圈，约28cm
RotateSpeedL = 0  # 取值范围:1代表转/秒，约28cm/s
RotateSpeedR = 0  # 取值范围:1代表转/秒，约28cm/s
TensionVL = 0  # 取值范围:单片机输入电压值，约0-3.3V
TensionVR = 0  # 取值范围:单片机输入电压值，约0-3.3V

def read():
    return [RevolutionsL, RevolutionsR, RotateSpeedL, RotateSpeedR, TensionVL, TensionVR]

class SensorThread(Thread):
    def run(self):
        global RevolutionsL
        global RevolutionsR
        global RotateSpeedL
        global RotateSpeedR
        global TensionVL
        global TensionVR
        self.ifdo = True
        try:
            SensorPort.open()
            print("SensorPort:" + SensorPort.portstr + " is successfully opend.")
            serialcmd = 'R'
            SensorPort.write(serialcmd.encode())
            while self.ifdo:
                # send data
                sleep(ProgramRunSpeed)
                serialcmd = 'H'
                SensorPort.write(serialcmd.encode())
                sleep(0.001)
                SensorData = SensorPort.read(25)
                if len(SensorData) == 24:
                    RevolutionsL, RevolutionsR, RotateSpeedL, RotateSpeedR, TensionVL, TensionVR = struct.unpack(
                        'ffffff', SensorData)
                    RevolutionsL = -RevolutionsL
                    RotateSpeedL = -RotateSpeedL
                    # print("Receive: L:%f R:%f L:%f R:%f L:%f R:%f" %(RevolutionsL,RevolutionsR,RotateSpeedL,RotateSpeedR,TensionVL,TensionVR))
            SensorPort.close()
            print("SensorPort:" + SensorPort.portstr + " is closed.")
        except Exception as ex:  # python 3
            # except Exception , ex:		#python 2
            print(ex)

    def reset(self):
        if SensorPort.isOpen():
            serialcmd = 'R'
            SensorPort.write(serialcmd.encode())
            sleep(ProgramRunSpeed * 3)

    def stop(self):
        # print('SensorThread is stopping ...')
        self.ifdo = False


if __name__ == '__main__':
    sensorthread = SensorThread()
    sensorthread.setDaemon(True)
    sensorthread.start()
    print('SensorThread is starting ...')

    while True:
        sleep(1)
        currenttime = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print(currenttime + ' ' + str('%.4f' % RevolutionsL) + ' ' + str('%.4f' % RevolutionsR) + ' ' +
              str('%.4f' % RotateSpeedL) + ' ' + str('%.4f' % RotateSpeedR) + ' ' +
              str('%.3f' % TensionVL) + ' ' + str('%.3f' % TensionVR) + ' '
              )

    print('I will stop it ...')
    sensorthread.stop()
    sensorthread.join()
