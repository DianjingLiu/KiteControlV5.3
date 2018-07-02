# -*- coding:utf-8 -*-  
import sys
import time
import os
import pygame
# import math
from time import sleep

import Sensor as Sensor
import Motor as Motor
import Tracker as Tracker
from View import *

import mmap
import contextlib
import struct
import numbers
import cv2
import numpy as np

import matplotlib.pyplot as plt
from pdb import set_trace

# 程序自检
def self_test(SensorThread, MotorThread):
    screen.fill((0,100,0))
    print_tipmessage(tip="Self Testing...", step=0)
    pygame.display.update()
    sleep(0.5)
    SensorThread.reset()
    if Sensor.RotateSpeedL != 0:
        print("The Left Motor is running!")
        return False
    if Sensor.RotateSpeedR != 0:
        print("The Right Motor is running!")
        return False

    Motor.MotorSetL = 0.1
    Motor.MotorSetR = 0.1
    sleep(0.4)
    # print(str(Sensor.RevolutionsL)+str(Sensor.RevolutionsL)+str(Sensor.RevolutionsR)+str(Sensor.RevolutionsR))
    if Sensor.RevolutionsL == 0:
        print("The connection of Left Motor or Encoder are broken!")
        return False
    if Sensor.RevolutionsL > -0.005:
        print("The connection of Left Motor or Encoder are inverse!")
        return False
    if Sensor.RevolutionsR == 0:
        print("The connection of Right Motor or Encoder are broken!")
        return False
    if Sensor.RevolutionsR > -0.005:
        print("The connection of Right Motor or Encoder are inverse!")
        return False
    Motor.MotorSetL = 0
    Motor.MotorSetR = 0
    sleep(0.5)
    print("Self Test is OK!")
    SensorThread.reset()
    # print(str(Sensor.RevolutionsL)+"  "+str(Sensor.RevolutionsR))
    return True

def DetectControl(MODE, state):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return -1 
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # 如果esc键按下，退出程序
                return -1
                print("<esc>")  
        elif event.type == pygame.JOYBUTTONDOWN:
            if event.button == 1:
                print('Manual Control.')
                MODE = 1
                state.reset_state()
            if event.button == 2:
                print('Auto Control.')
                MODE = 2
            """
            if event.button == 3:
                print('Kernel update on.')
                Tracker.UPDATE_KERNEL = True
            if event.button == 0:
                print('Kernel update off.')
                Tracker.UPDATE_KERNEL = False
			"""
    return MODE
    """
    # 检测键盘以及Joystick按键并响应
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running_flag = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # 如果esc键按下，退出程序
                running_flag = False
                print("<esc>")
            if event.key == pygame.K_RETURN:  # 如果回车键按下，复位设备
                MotorThread.reset()
                SensorThread.reset()
                SetLengthStartL = 0
                SetLengthStartR = 0
                print('<reset>')
            if event.key == pygame.K_COMMA:  # 如果<键按下
                scale = scale - 0.1
                if scale < 0:
                    scale = 0
                print('scale: ' + str(scale * 100) + '%')
            if event.key == pygame.K_PERIOD:  # 如果>键按下
                scale = scale + 0.1
                if scale > max_scale:
                    scale = max_scale
                print('scale: ' + str(scale * 100) + '%')
            if event.key == pygame.K_UP:
                Wheel_button = 1
            if event.key == pygame.K_DOWN:
                Wheel_button = -1
            if event.key == pygame.K_LEFT:
                L1_button = 1
            if event.key == pygame.K_RIGHT:
                R1_button = 1
        elif event.type == pygame.KEYUP:
            if event.key==pygame.K_UP:
                Wheel_button = 0
            if event.key==pygame.K_DOWN:
                Wheel_button = 0
            if event.key==pygame.K_LEFT:
                L1_button = 0
            if event.key==pygame.K_RIGHT:
                R1_button = 0
        elif event.type == pygame.JOYBUTTONDOWN:
            # 检测joystick并响应
            if event.button == 11:  # OPTIONS键
                running_flag = False
                print("<esc>")
            if event.button == 10:  # SHAPE键，复位设备
                MotorThread.reset()
                SensorThread.reset()
                SetLengthStartL = 0
                SetLengthStartR = 0
                print('<reset>')
            if event.button == 1:  # 圆圈键（右键）
                scale = scale - 0.1
                if scale < 0:
                    scale = 0
                print('scale: ' + str(scale * 100) + '%')
            if event.button == 3:  # 三角键（上键）
                scale = scale + 0.1
                if scale > max_scale:
                    scale = max_scale
                print('scale: ' + str(scale * 100) + '%')
            if event.button == 0:  # 左侧拨键
                left_button = 1
            if event.button == 2:  # 右侧拨键
                right_button = 1
            if event.button == 4:  # L1
                L1_button = 1
            if event.button == 5:  # R1
                R1_button = 1
        elif event.type == pygame.JOYBUTTONUP:
            if event.button == 0:  # 左侧拨键
                left_button = 0
            if event.button == 2:  # 右侧拨键
                right_button = 0
            if event.button == 4:  # L1
                L1_button = 0
            if event.button == 5:  # R1
                R1_button = 0
    """

def WheelControl(joystick):
    ###################################################
    # TODO: add controlling of single motor
    '''
    # for event in pygame.event.get():
    if event.type == pygame.JOYBUTTONDOWN:
        if event.button == 0:  # 左侧拨键
            left_button = 1
        if event.button == 2:  # 右侧拨键
            right_button = 1
    elif event.type == pygame.JOYBUTTONUP:
        if event.button == 0:  # 左侧拨键
            left_button = 0
            controller.reset()
        if event.button == 2:  # 右侧拨键
            right_button = 0 
            controller.reset()      
    '''
    #####################################################

    Wheel_button = joystick.get_axis(0)  # 方向盘
    Wheel_button = round(Wheel_button, 2)
    Accelerator = joystick.get_axis(2)  # 
    Accelerator = round(Accelerator, 2)
    #controller.update(Wheel_button)
    return Wheel_button, Accelerator

class state:
    def __init__(self, n_input=20, clean_s=True):
        """
        Manage state and joystick.
        """
        self._max_length = n_input
        self._clean_s = clean_s # If True, the state will be cleared when tracking failed.
        #self.joystick = 0
        self._ratio_a = 18.0 # Joystick step size is 1/ratio_a. Action difference = 1 represents: joystick += 1/ratio_a
                            # This value should be equal to variable 'ratio_a' in 'make_memory.py'
        self._s = [] # Initialize state

    def set_n_input(self,n):
        self._max_length = n # change state length when no tension input

    def reset_state(self):
        self._s = []

    def update(self,log):
        # Process log from 
        # Params:
        #   log: [ok, center_loc[0], center_loc[1], angle, 
        #            Sensor.RevolutionsL, Sensor.RevolutionsR, 
        #            Sensor.RotateSpeedL, Sensor.RotateSpeedR, 
        #            Sensor.TensionVL, Sensor.TensionVR
        #            Wheel_button #(For RL)
        #           ]
        
        if log[0]==0:
            # If tracking failed: delete state
            if self._clean_s:
                self.reset_state()
        else:
            # If tracking succeed: 
            record = np.array(log[1::])
            # Process log: x=(x-700)*0.001. phi=phi*np.pi/180
            record[0:2] = (record[0:2]-700)*0.001
            record[2] = record[2]*np.pi/180.0

            self._s = np.concatenate((self._s, record))
            n_delete = len(self._s) - self._max_length
            #if len(self._s) > self._max_length:
            if n_delete > 0:
                self._s = np.delete(self._s, range(n_delete), 0)
        
        return self.get_state()

    def get_state(self):
        """
        Return kite state. 
        If state is not complete, return None.
        """
        if len(self._s)==self._max_length:
            return self._s[np.newaxis,:]
        else:
            return None
    def update_joystick(self, action, joystick):
        step = (action - 2.0) / self._ratio_a
        joystick = joystick + step
        # clip the value within [-1,1]
        return np.clip(joystick, -1, 1)

class motor_control:
    """
    Control motor PWM according to joystick wheel position.
    """
    def __init__(self):
        self.SetLengthStartL=0 # Left motor balanced position (when Wheel_button=0). Unit: mm
        self.SetLengthStartR=0 # Right motor balanced position (when Wheel_button=0). Unit: mm
        self.SetLengthL=0
        self.SetLengthR=0
        # Motor params
        self.max_diff = (280.0 / 2) * 2 # Max length difference 
        self.max_reel_speed = 0.08
        self.factor = 280.0 # Length change 280mm for each circle

    def reset(self):
        # Reset controller
        self.SetLengthStartL=0
        self.SetLengthStartR=0 

    def update(self,Wheel_button, Accelerator=0):
        # Adjust kite line balance position
        self.SetLengthStartL = self.SetLengthStartL - Accelerator * self.max_reel_speed
        self.SetLengthStartR = self.SetLengthStartR - Accelerator * self.max_reel_speed

        # Differential mode
        self.SetLengthL = self.SetLengthStartL - Wheel_button * self.max_diff / self.factor# l1 = 0 - Joystick
        self.SetLengthR = self.SetLengthStartR + Wheel_button * self.max_diff / self.factor
        return self.set_motor()
    
    def set_motor(self):
        # 根据设定长度和传感器实际长度以及速度计算给定值
        # p=1.5             #1.5调好
        # i=0.02
        # p=2               #2
        # i=0.15
        p = 3  # 3
        i = 0.24
    
        pl = p  # 调节稳态精度，误差小于1cm
        il = i  # 调节到达稳态时的振荡
        pr = p  # 调节稳态精度，误差小于1cm
        ir = i  # 调节到达稳态时的振荡
        # 调节两个参数对应左侧电机长度误差反馈和速度反馈
        motor_l = pl * (Sensor.RevolutionsL - self.SetLengthL) + il * Sensor.RotateSpeedL
        # print('SetLengthL: '+str(SetLengthL)+'  MotorSetL: '+str(MotorSetL))
    
        # 调节两个参数对应右侧电机长度误差反馈和速度反馈
        motor_r = pr * (Sensor.RevolutionsR - self.SetLengthR) + ir * Sensor.RotateSpeedR
        # print('SetLengthR: '+str(SetLengthR)+'  MotorSetR: '+str(MotorSetR))
    
        motor_l = np.clip(motor_l, -1, 1)
        motor_r = np.clip(motor_r, -1, 1)

        # Set Motor Params
        Motor.MotorSetL = motor_l
        Motor.MotorSetR = motor_r
        return


if __name__ == '__main__':
    #from Kite_Tracking_angle_detection import config
    #from Kite_Tracking_angle_detection import MLP
    import matplotlib.image as mpimg
    init_frame = mpimg.imread("bg.bmp")

    from Kite_Tracking_angle_detection.interface import *
    tracker = Interface()
    for _ in range(INIT_FRAMES_NUM):
        tracker.init_tracker(init_frame)
    state=state()
    controller=motor_control()
    
    # Import RL
    from Kite_agent.kite_agent import *
    
    """
    # Record timestamp
    timestamp = round(time.time(),4)
    # Convert to datetime:
    import datetime
    #print(datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f'))

    # Get record from sensor and tracker
    log = get_sensor()
    frame = get_img()
    ok, bbox, angle, center_loc = tracker.update(frame, verbose=False)
    if ok:
        log = [ok, center_loc[0], center_loc[1], angle] + log + [state.joystick]
    else:
        log = [ok, None, None, None] + log + [state.joystick]

    # Update state
    # Choose action and update joystick
    s = state.update(log)
    J = None
    if s is not None:
        a = RL.choose_action(s)
        Wheel_button = state.update_joystick(a)
        controller.update(Wheel_button)

    # Add timestamp to log, and save to file
    #log = [timestamp] + log 
    """

    # DEBUG: test run time
    LogFile = open('test.txt', 'w')
    LogFile.write('# Timestamp ' + 'TrackSucceed ' + 'KiteLoc1 ' + 'KiteLoc2 ' + 'KiteAngle '
        + 'RevolutionsL ' + 'RevolutionsR '
        + 'RotateSpeedL ' + 'RotateSpeedR ' 
        + 'TensionVL ' + 'TensionVR ' 
        + 'JoyStickWheel ' 
        #+ 'Action ' + 'MotorSetL ' + 'MotorSetR ' 
        + '\n')
    ok=False
    while not ok:
        frame = get_img()
        ok, bbox, angle, center_loc = tracker.update(frame, verbose=False)

    frame = get_img()
    t1 = time.time()
    for _ in range(10):
        timestamp = round(time.time(),4)
        log=[]
        log = get_sensor()
        _f = get_img()
        ok, bbox, angle, center_loc = tracker.update(frame, verbose=False)
        if ok:
            log = [ok, center_loc[0], center_loc[1], angle] + log + [state.joystick]
        else:
            log = [ok, None, None, None] + log + [state.joystick]
        s = state.update(log) # log is also updated
        if s is not None:
            a = RL.choose_action(s)
            Wheel_button = state.update_joystick(a)
            controller.update(Wheel_button)
        log = [timestamp] + log
        #print(log)
        LogFile.write(', '.join(str(e) for e in log)+'\n')
    print('Run time: ' + str(time.time()-t1))
    LogFile.close()
