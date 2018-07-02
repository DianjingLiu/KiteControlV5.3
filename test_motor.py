# -*- coding:utf-8 -*-  
import sys
import time
import os
import datetime
# import pygame
# import math
from time import sleep

import Sensor as Sensor
import Motor as Motor
import Tracker as Tracker

from View import *
from Control_utils import *
from Kite_agent.kite_agent import *  

# Import AI
AI_MODEL = 1 # AI modal: 0 -- ordinary
             #           1 -- no tension input
RL = build_AI(AI_MODEL)
#from Kite_Tracking_angle_detection.interface import *


MODE = 1 # Control mode: 0 -- Exit control 
         #               1 -- Manual control
         #               2 -- AI control
ProgramRunSpeed = 0.04  # 刷新频率，间隔0.04秒,每秒25帧

# Initialize Sensor parameters
Sensor.ProgramRunSpeed = ProgramRunSpeed  # 刷新频率，间隔0.1秒
Sensor.RevolutionsL = 0  # 取值范围:1代表一圈，约28cm
Sensor.RevolutionsR = 0  # 取值范围:1代表一圈，约28cm
Sensor.RotateSpeedL = 0  # 取值范围:1代表转/秒，约28cm/s
Sensor.RotateSpeedR = 0  # 取值范围:1代表转/秒，约28cm/s
Sensor.TensionVL = 0  # 取值范围:单片机输入电压值，约0-3.3V
Sensor.TensionVR = 0  # 取值范围:单片机输入电压值，约0-3.3V

# Initialize Motor Parameters
Motor.ProgramRunSpeed = ProgramRunSpeed  # 刷新频率，间隔0.1秒
Motor.MotorSetL = 0  # 取值范围:电机转速，-1--+1,对应最大转速，约1.5圈/秒
Motor.MotorSetR = 0  # 取值范围:电机转速，-1--+1,对应最大转速，约1.5圈/秒

# Initialize Tracker Parameters
Tracker.ProgramRunSpeed = ProgramRunSpeed
Tracker.INIT_FRAMES_NUM = 3
Tracker.READ_FROM_FILE = False # For debug. Read image from file

state = state()
if AI_MODEL == 1:
    state.set_n_input(16) # change state length to 16 when no tension input
controller = motor_control()

# Log file path
CurrentTime = time.time()
CurrentTime = datetime.datetime.fromtimestamp(CurrentTime).strftime('%Y-%m-%d %H-%M-%S')
LOG_PATH = './log/log_' + CurrentTime + '.txt'



if __name__ == '__main__':
    # 启动Sensor通讯线程
    Sensor.SensorPort.port = 'COM6'
    Sensor.ProgramRunSpeed = ProgramRunSpeed
    SensorThread = Sensor.SensorThread()
    SensorThread.setDaemon(True)
    #SensorThread.start()

    # 启动Motor通讯线程
    Motor.MotorPort.port = 'COM5'
    Motor.ProgramRunSpeed = ProgramRunSpeed
    MotorThread = Motor.MotorThread()
    MotorThread.setDaemon(True)
    #MotorThread.start()

    # Start Tracker thread
    TrackerThread = Tracker.TrackerThread()
    TrackerThread.setDaemon(True)
    TrackerThread.start() # Here the kite tracker is initialized by camera images

    # 初始化显示窗口
    x = 50
    y = 50
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)
    pygame.init()
    pygame.display.set_caption("Control Demo")
    screen = pygame.display.set_mode((800, 600), 0, 32)
    
    # 初始化joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    #joystick=None

    # Initialize tracker
    # import matplotlib.image as mpimg
    # init_frame = mpimg.imread("bg.bmp")
    # tracker.init_tracker(init_frame)

    Wheel_button = state.getWheel_button()

    # 打开记录文件
    LogFile = open(LOG_PATH, 'w')
    if AI_MODEL==0:
        LogFile.write('# Timestamp ' + 'TrackSucceed ' + 'KiteLoc1 ' + 'KiteLoc2 ' + 'KiteAngle '
            + 'RevolutionsL ' + 'RevolutionsR '
            + 'RotateSpeedL ' + 'RotateSpeedR ' 
            + 'TensionVL ' + 'TensionVR ' 
            + 'JoyStickWheel ' 
            #+ 'Action ' + 'MotorSetL ' + 'MotorSetR ' 
            + '\n')
    elif AI_MODEL==1:
        LogFile.write('# Timestamp ' + 'TrackSucceed ' + 'KiteLoc1 ' + 'KiteLoc2 ' + 'KiteAngle '
            + 'RevolutionsL ' + 'RevolutionsR '
            + 'RotateSpeedL ' + 'RotateSpeedR ' 
            + 'JoyStickWheel '+ '\n')

    # 刷新频率计数
    start_time = time.time()
    end_time = time.time()
    refresh_rate = 0
    refresh_count = 0
    # 设备自检
    #running_flag = self_test(SensorThread)
    running_flag = True
    """
    Wheel_button = 0
    L1_button=0
    R1_button=0
    left_button=0
    right_button=0
    SerialNum=0
    """
    while running_flag:
        ct_start = time.time()
        # Record timestamp
        timestamp = time.time()
        # Get record from sensor and tracker
        # log = [ok, x, y, phi, l1, l2, v1, v2, T1, T2, J]
        log1 = Tracker.read()
        log2 = Sensor.read()
        if AI_MODEL == 1:
            del log2[-2:] # No tension
        log = log1+log2+[Wheel_button]
        """
        frame = get_img()
        test_time = time.time()
        ok, bbox, angle, center_loc = tracker.update(frame, verbose=False)
        if ok:
            log = [ok, center_loc[0], center_loc[1], angle] + log + [state.joystick]
        else:
            log = [ok, None, None, None] + log + [state.joystick]
        """
        MODE = DetectControl(MODE,state)
        # Update wheel button
        if MODE == 0:
            running_flag = False
        elif MODE == 1:
            Wheel_button = WheelControl(joystick, controller)
        elif MODE == 2:
            # Update state
            s = state.update(log)
            # Choose action and update joystick
            if s is not None:
                a = RL.choose_action(s)
                Wheel_button = state.update_joystick(a) # Wheel_button
                controller.update(Wheel_button)
        else:
            print('Wrong MODE value: {}'.format(MODE))
            running_flag = False

        test_time = time.time()-timestamp
        # Write log to file
        log = [timestamp] + log
        LogFile.write(', '.join(str(e) for e in log)+'\n')


        # clean
        screen.fill((0, 100, 0))
        # 在屏幕上显示相关参数
        print_all(Sensor.RevolutionsL, Sensor.RevolutionsR, Sensor.RotateSpeedL, Sensor.RotateSpeedR,
                  Sensor.TensionVL,
                  Sensor.TensionVR)
        print_Operate(Wheel_button)
        # 显示刷新频率
        print_refresh(refresh_rate)
        # 更新屏幕
        pygame.display.update()

        # 统计刷新频率
        refresh_count += 1
        end_time = time.time()
        if (end_time - start_time) >= 1:
            start_time = end_time
            refresh_rate = refresh_count
            refresh_count = 0
        """
        # 保存记录到文件
        local_time = time.localtime(ct_start)
        data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        data_secs = (ct_start - int(ct_start)) * 1000
        time_stamp = "%s.%03d" % (data_head, data_secs)
        #CurrentTime = time.strftime('%Y-%m-%d %H-%M-%S ', time.localtime(time.time()))
        SerialNum+=1
        LogFile.write(str(SerialNum)+' ' + 
            time_stamp + ' ' + str('%.4f' % Sensor.RevolutionsL) + ' ' + str('%.4f' % Sensor.RevolutionsR) + ' ' +
            str('%.4f' % Sensor.RotateSpeedL) + ' ' + str('%.4f' % Sensor.RotateSpeedR) + ' ' +
            str('%.3f' % Sensor.TensionVL) + ' ' + str('%.3f' % Sensor.TensionVR) + ' ' +
            str('%.4f' % Wheel_button) + ' ' + str('%.4f' % ((L1_button <<3)+(R1_button<<2)+(left_button<<1)+(right_button<<0))) + ' ' +
            str('%.4f' % Motor.MotorSetL) + ' ' + str('%.4f' % Motor.MotorSetR) + '\n')
        """

        ct_end = time.time()
        tot_time = ct_end-ct_start
        #print('tot_time: {:1f}ms. track time: {:1f}ms. rest: {:1f}ms.'.format(tot_time*1000, test_time*1000, (tot_time-test_time)*1000) )
        # Check log
        print(log, Motor.MotorSetL, Motor.MotorSetR)
         # 等待0.1秒       
        if ProgramRunSpeed-0.001>(ct_end-ct_start):
            sleep(ProgramRunSpeed-0.001-(ct_end-ct_start)) 


    print('Program will stop ...')
    # 关闭并退出
    LogFile.close()
    # 停止电机
    #MotorThread.stop()
    #SensorThread.stop()
    TrackerThread.stop()
    #MotorThread.join()
    #SensorThread.join()
    TrackerThread.join()
    print_close()
    print('End')
    sys.exit(0)
    pass
