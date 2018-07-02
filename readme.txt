Operate instruction:
1.Joystick must be connected before run the program. if not it will return a error:pygame.error: Invalid joystick device number
2.After started the program will init the Secsor com port and the Motor com port
3.the program will run the self_test, which will control the motor to turn a bit, and test if the motor and the encoder is OK, if not it will return a error
4.the system can be control by keyboard or Joystick.
  <ese> or <OPTIONS> will Exit the program
  <enter> or <SHAPE> will reset the Motor and the Sensor, the Motor will be stoped and the Sensor data will be clear
  < < > and < > > on keyboard or <right> and <up> on Joystick will change the motor response speed
5.when the L1_button on Joystick is pushdown, the wheel only control the left Motor run
  when the R1_button on Joystick is pushdown, the wheel only control the right Motor run
  when both of the L1_button and R1_button on Joystick is pushdown, the wheel only control the both the left and right Motor run
  when none of the L1_button and R1_button on joystick is pushdown:
      the system will run on work mode.
      the Joystick wheel will control the difference of the left and right line, so the direction of the kite in sky will be changed.
      if the left plectrum on joystick is pushed, the both of the line will be shorted
      if the right plectrum on joystick is pushed, the both of the line will be extenstion.

Version 5.1:
Separate tracker as an individual thread.

Version 5.2: 
Add supervised training AI. Wheel_button is managed by main program ('Control.py').
Add kernel updating function

Version 5.3:
Add AI discription in log file.

Optimized code in 'Control_utils.py':
Class state:
	Add attribute '_clean_s' (default True). If '_clean_s' is False, the state is reset when tracking failed. 
	Attribute '_s' (state) is initialized as empty list.
class motor_control:
	Add accelerator control. Can reel in/out simutaneously.

Kite_Tracking_speedup:
Updated kernel image is rotated to point upward.