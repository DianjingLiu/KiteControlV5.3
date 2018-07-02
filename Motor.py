#  -*- coding:utf-8 -*-  
import serial
# import struct
from threading import *
from time import sleep

# 创建电机驱动
MotorPort = serial.Serial()
MotorPort.port = 'COM5'
MotorPort.baudrate = 115200
MotorPort.parity = 'N'
MotorPort.bytesize = 8
MotorPort.stopbits = 1
MotorPort.timeout = 0
MotorPort.write_timeout = 0

ProgramRunSpeed = 0.1  # 刷新频率，间隔0.1秒

MotorSetL = 1  # 取值范围:电机转速，-1--+1,对应最大转速，约1.5圈/秒
MotorSetR = 1  # 取值范围:电机转速，-1--+1,对应最大转速，约1.5圈/秒


class MotorThread(Thread):
    def run(self):
        global MotorSetL
        global MotorSetR
        self.ifdo = True
        try:
            MotorPort.open()
            print("MotorPort:" + MotorPort.portstr + " is successfully opend.")

            # ramprate = 1900
            ramprate = 1000
            cmd1 = 'R1:' + str(ramprate) + '\r\n'
            cmd2 = 'R2:' + str(ramprate) + '\r\n'
            MotorPort.write(cmd1.encode())
            MotorPort.write(cmd2.encode())

            while self.ifdo:
                # send data
                sleep(ProgramRunSpeed)

                move1 = 'M1:' + str(int(MotorSetL * 2047)) + '\r\n'
                move2 = 'M2:' + str(int(MotorSetR * 2047)) + '\r\n'

                MotorPort.write(move1.encode())
                MotorPort.write(move2.encode())
                
            # stop the motor
            move1 = 'M1:0\r\n'
            move2 = 'M2:0\r\n'
            MotorPort.write(move1.encode())
            MotorPort.write(move2.encode())
            sleep(ProgramRunSpeed)
            MotorPort.close()
            print("MotorPort:" + MotorPort.portstr + " is closed.")
        except Exception as ex:  # python 3
            # except Exception , ex:		#python 2
            print(ex)

    def reset(self):
        if MotorPort.isOpen():
            move1 = 'M1:0\r\n'
            move2 = 'M2:0\r\n'
            MotorPort.write(move1.encode())
            MotorPort.write(move2.encode())
            sleep(ProgramRunSpeed * 3)

    def stop(self):
        # print('MotorThread is stopping ...')
        self.ifdo = False


if __name__ == '__main__':
    motorthread = MotorThread()
    motorthread.setDaemon(True)
    motorthread.start()
    print('MotorThread is starting ...')

    sleep(0.5)

    print('I will stop it ...')
    motorthread.stop()
    motorthread.join()
