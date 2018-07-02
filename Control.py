# -*- coding:utf-8 -*-  
"""
Control V5.2: Add supervised training AI. Wheel_button is managed by main program ('Control.py').
Control V5.3: Code Optimized. class state in 'Control_utils.py' is better organized. 
                Add AI discription in log file.
                Add accelerator control.
"""
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
#from Kite_Tracking_angle_detection.interface import *

from View import *
from Control_utils import *
from Kite_agent.kite_agent import *  


# Import AI
AI_MODEL = 15 # AI model: 
             # 0 -- RL. rule 2: point to target point (100 below center)
             # 1 -- RL. rule 3: close to target point (100 below center) Tend to go circle
             # 2 -- RL. rule 4: rule 2 + 3          Tend to go circle
             # 3 -- RL. rule 5: opposite to rule 2. 
             # 4 -- RL. rule 6: -exp(distance).
             # 5 -- RL. rule 2: No tension input TRY
             # 6 -- RL. rule 3: No tension input  Tend to go circle
             # 7 -- RL. rule 4: No tension input  Tend to go circle
             # 8 -- RL. rule 5: No tension input
             # 9 -- RL. rule 6: No tension input
             # 10-- DNN. input 7 records
             # 11-- DNN. input 7 records, No tension input
             # 12-- DNN. input 20 records
             # 13-- DNN. input 20 records, No tension input
             # 14-- DNN. Clone FLC. input 7 records, No tension input'.
             # 15-- Fuzzy logic control.


AI = build_AI(AI_MODEL)
processor = state(N_INPUT[AI_MODEL], clean_s=True)

controller = motor_control()


MODE = 1 # Control mode: -1 -- Exit control 
         #               0 -- Reset device
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
Tracker.INIT_FRAMES_NUM = 10
Tracker.READ_FROM_FILE = False # For debug. Read image from file
Tracker.DISPLAY = True
Tracker.IMAGE_PATH = "F:\\GrapV1.13\\Dest\\*.bmp"

# Log file path
CurrentTime = time.time()
CurrentTime = datetime.datetime.fromtimestamp(CurrentTime).strftime('%Y-%m-%d %H-%M-%S')
LOG_PATH = './log/log_' + CurrentTime + '_AI{:d}.txt'.format(AI_MODEL)



if __name__ == '__main__':
    # 启动Sensor通讯线程
    Sensor.SensorPort.port = 'COM6'
    Sensor.ProgramRunSpeed = ProgramRunSpeed
    SensorThread = Sensor.SensorThread()
    SensorThread.setDaemon(True)
    SensorThread.start()

    # 启动Motor通讯线程
    Motor.MotorPort.port = 'COM5'
    Motor.ProgramRunSpeed = ProgramRunSpeed
    MotorThread = Motor.MotorThread()
    MotorThread.setDaemon(True)
    MotorThread.start()


    # Start Tracker thread
    TrackerThread = Tracker.TrackerThread()
    TrackerThread.init_tracker()
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

    Wheel_button = 0
    Accelerator = 0

    # 打开记录文件
    LogFile = open(LOG_PATH, 'w')
    LogFile.write(AI_DISCRIPTION[AI_MODEL] + '\n')
    LogFile.write('# Timestamp ' + 'ControlMode ' + 'TrackSucceed ' + 'KiteLoc1 ' + 'KiteLoc2 ' + 'KiteAngle '
        + 'RevolutionsL ' + 'RevolutionsR '
        + 'RotateSpeedL ' + 'RotateSpeedR '
        + '{}'.format('' if AI_NT[AI_MODEL] else 'TensionVL TensionVR ') 
        #+ 'TensionVL ' + 'TensionVR ' 
        + 'JoyStickWheel ' #+ 'JoystickAccelerator ' 
        #+ 'Action ' + 'MotorSetL ' + 'MotorSetR ' 
        + '\n')

    # 刷新频率计数
    start_time = time.time()
    end_time = time.time()
    refresh_rate = 0
    refresh_count = 0
    # 设备自检
    running_flag = self_test(SensorThread,MotorThread)
    #running_flag = True

    while running_flag:
        ct_start = time.time()
        # Record timestamp
        timestamp = time.time()
        # Get record from sensor and tracker
        # log = [ok, x, y, phi, l1, l2, v1, v2, T1, T2, J]
        log1 = Tracker.read() # [ok, x, y, phi]
        log2 = Sensor.read()  # [l1, l2, v1, v2, T1, T2]
        if AI_NT[AI_MODEL] == 1:
            del log2[-2:] # No tension
        #log = log1+log2+[Wheel_button, Accelerator]
        log = log1+log2+[Wheel_button]

        MODE = DetectControl(MODE,processor)
        # Update wheel button
        if MODE == -1:
            running_flag = False
        elif MODE == 0:
            MotorThread.reset()
            SensorThread.reset()
            controller.reset()
            #processor.reset_state()
            print('<reset>')
        elif MODE == 1:
            Wheel_button, Accelerator = WheelControl(joystick)
        elif MODE == 2:
            # Fuzzy logic control
            if MODEL_PATH[AI_MODEL] is None:
                # Update state
                s = processor.update(log)
                if log[0]==1:
                    if s is not None:
                        Wheel_button = AI.control(s)
            # RL
            elif AI_RL[AI_MODEL]:
                # Update state
                s = processor.update(log)
                # Choose action and update joystick
                if s is not None:
                    a = AI.choose_action(s)
                    Wheel_button = processor.update_joystick(a, Wheel_button) # Wheel_button
            # DNN
            else:
                s = processor.update(log[0:-1])
                if s is not None:
                    Wheel_button = AI.predict(s)
                    Wheel_button = Wheel_button[0,0]
            if s is None:
                Wheel_button = 0
        else:
            print('Wrong MODE value: {}'.format(MODE))
            running_flag = False
        # Send control command to motor
        controller.update(Wheel_button, Accelerator)

        # Write log to file
        log = [timestamp, MODE] + log
        LogFile.write(', '.join(str(e) for e in log)+'\n')
        # Check log
        print(log)


        # clean
        screen.fill((0, 100, 0))
        # 在屏幕上显示相关参数
        print_all(Sensor.RevolutionsL, Sensor.RevolutionsR, Sensor.RotateSpeedL, Sensor.RotateSpeedR,
                  Sensor.TensionVL,
                  Sensor.TensionVR)
        print_Operate(Wheel_button)
        # 显示刷新频率
        print_refresh(refresh_rate)
        # Show MODE
        print_mode(MODE, AI_MODEL)
        # Show tracking time
        print_track_time(Tracker.RUNTIME, log1[0])        
        # 更新屏幕
        pygame.display.update()

        # 统计刷新频率
        refresh_count += 1
        end_time = time.time()
        if (end_time - start_time) >= 1:
            start_time = end_time
            refresh_rate = refresh_count
            refresh_count = 0
        ct_end = time.time()
        #test_tot_time = ct_end-ct_start
        #print('tot_time: {:1f}ms. track time: {:1f}ms. rest: {:1f}ms.'.format(test_tot_time*1000, test_time*1000, (test_tot_time-test_time)*1000) )
         # 等待0.1秒       
        if ProgramRunSpeed-0.001>(ct_end-ct_start):
            sleep(ProgramRunSpeed-0.001-(ct_end-ct_start)) 


    print('Program will stop ...')
    # 关闭并退出
    LogFile.close()
    # 停止电机
    MotorThread.stop()
    SensorThread.stop()
    TrackerThread.stop()
    MotorThread.join()
    SensorThread.join()
    TrackerThread.join()
    print_close()
    print('End')
    sys.exit(0)
    pass
