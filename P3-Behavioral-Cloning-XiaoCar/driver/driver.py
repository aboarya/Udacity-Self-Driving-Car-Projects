#!/usr/bin/env python3
# coding: Latin-1
"""
This module acts as a driver for the robot.  It provides an joystick interface and
acts as entry point to data collection.

Creating an instance of the Driver class will setup a single Joystick.

Additionally it makes use of the StreamProcessor and ImageCapture classes
to caputre images as the robot dives and labels each image with the
left and right motor power levels.
"""
import time
import os
import pickle
import sys
import threading

import cv2
import picamera
import picamera.array
import pygame

from . import controller
from model import save_data
from model import predict

sys.stdout = sys.stderr

# Global values
global lastFrame
global lockFrame
global camera
global processor
global running
global DATA
global CONTROL

# Image stream processing thread
class StreamProcessor(threading.Thread):

    def __init__(self, camera):
        super(StreamProcessor, self).__init__()
        self.stream = picamera.array.PiRGBArray(camera)
        self.event = threading.Event()
        self.terminated = False
        self.start()
        self.begin = 0


    def run(self):
        global lastFrame
        global lockFrame
        global DATA
        global CONTROL
        print('in process stream run')
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    # Read the image and save globally
                    self.stream.seek(0)
                    retval, thisFrame = cv2.imencode('.jpg', self.stream.array)
                    lockFrame.acquire()
                    if 'driveLeft' in CONTROL:
                        DATA[str(time.time())] = {
                            'img' : thisFrame,
                            'left': CONTROL['driveLeft'],
                            'right': CONTROL['driveRight']}
                    lastFrame = thisFrame
                    lockFrame.release()
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()


# Image capture thread
class ImageCapture(threading.Thread):

    def __init__(self):

        super(ImageCapture, self).__init__()
        self.start()

    def run(self):
        global camera
        global processor
        print('Start the stream using the video port')
        camera.capture_sequence(self.TriggerStream(), format='bgr', use_video_port=True)
        print('Terminating camera processing...')
        processor.terminated = True
        processor.join()
        print('Processing terminated.')

    # Stream delegation loop
    def TriggerStream(self):
        print('in trigger stream')
        global running
        while running:
            if processor.event.is_set():
                time.sleep(0.25)
            else:
                yield processor.stream
                processor.event.set()


class Driver(object):

    # Settings for the joystick
    axisUpDown = 1                          # Joystick axis to read for up / down position
    axisUpDownInverted = True               # Set this to True if up and down appear to be swapped
    axisLeftRight = 2                       # Joystick axis to read for left / right position
    axisLeftRightInverted = True            # Set this to True if left and right appear to be swapped
    buttonResetEpo = 3                      # Joystick button number to perform an EPO reset (Start)
    buttonSlow = 8                          # Joystick button number for driving slowly whilst held (L2)
    slowFactor = 0.5                        # Speed to slow to when the drive slowly button is held, e.g. 0.5 would be half speed
    buttonFastTurn = 9                      # Joystick button number for turning fast (R2)
    buttonSave = 14                         # Joystick button number for turning fast (R2)
    interval = 0.00                         # Time between updates in seconds, smaller responds faster but uses more processor time
    hadEvent = False
    upDown = 0.0
    leftRight = 0.0

    def __init__(self, autonomous=False):

        self.setup_driver()
        self.autonomous = autonomous
        if not self.autonomous:
            self.setup_joystick()

    def setup_joystick(self):
        # Setup pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy" # Removes the need to have a GUI window
        pygame.init()
        pygame.joystick.init()
        pygame.display.set_mode((1,1))
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        # print('Setup the watchdog')
        # watchdog = Watchdog()

    def setup_driver(self):
        # Setup the PicoBorg Reverse
        self.PBR = controller.PicoBorgRev()
        self.PBR.Init()
        if not self.PBR.foundChip:
            boards = controller.ScanForPicoBorgReverse()
            if len(boards) == 0:
                print('No PicoBorg Reverse found, check you are attached :)')
            else:
                print('No PicoBorg Reverse at address %02X, but we did find boards:' % (self.PBR.i2cAddress))
                for board in boards:
                    print('    %02X (%d)' % (board, board))
                print('If you need to change the I²C address change the setup line so it is correct, e.g.')
                print('self.PBR.i2cAddress = 0x%02X' % (boards[0]))
            sys.exit()

        self.PBR.ResetEpo()

        return self.PBR

    def had_event(self, event):
        if event.type == pygame.QUIT:
            # User exit
            running = False
        elif event.type == pygame.JOYBUTTONDOWN:
            # A button on the joystick just got pushed down
            return True                    
        elif event.type == pygame.JOYAXISMOTION:
            # A joystick has been moved
            return True

        return False

    def set_axis(self):
        # Read axis positions (-1 to +1)
        if self.axisUpDownInverted:
            upDown = -self.joystick.get_axis(self.axisUpDown)
        else:
            upDown = self.joystick.get_axis(self.axisUpDown)

        if self.axisLeftRightInverted:
            leftRight = -self.joystick.get_axis(self.axisLeftRight)
        else:
            leftRight = self.joystick.get_axis(self.axisLeftRight)

        return (upDown, leftRight)

    def steer(self, leftRight, upDown):
        # Apply steering speeds
        if not self.joystick.get_button(self.buttonFastTurn):
            leftRight *= 0.5
            # Determine the drive power levels
            CONTROL['driveLeft'] = -upDown
            CONTROL['driveRight'] = -upDown

        if leftRight < -0.05:
            # Turning left
            CONTROL['driveLeft'] *= 1.0 + (2.0 * leftRight)
        elif leftRight > 0.05:
            # Turning right
            CONTROL['driveRight'] *= 1.0 - (2.0 * leftRight)

        if self.joystick.get_button(self.buttonSlow):
            CONTROL['driveLeft'] *= slowFactor
            CONTROL['driveRight'] *= slowFactor

        if self.joystick.get_button(6):
            self._pickle()

    def _pickle(self):
        print("saving to file...")
        running = False
        processor.terminated = True
        self.PBR.MotorsOff()
        save_data(DATA, file_path='robot-day-short.p')
        sys.exit(0)

    def drive_joystick(self):
        # Get the latest events from the system
        self.hadEvent = False
        events = pygame.event.get()
        # Handle each event individually
        for event in events:
            if self.had_event(event):
                (upDown, leftRight) = self.set_axis()
                self.steer(leftRight, upDown)

                # Check for button presses
                if self.joystick.get_button(self.buttonResetEpo):
                    self.PBR.ResetEpo()

                # Set the motors to the new speeds
                self.PBR.SetMotor1(CONTROL['driveLeft'])
                self.PBR.SetMotor2(CONTROL['driveRight'])

    def drive_autonoumous(self):
        if lastFrame is not None:

            data = list(predict(lastFrame))[0]
            print(data)
            # Set the motors to the new speeds
            left, right = data[0], data[1]
            self.PBR.SetMotor1(left)
            self.PBR.SetMotor2(right)

    def drive(self):
        # Loop indefinitely
        global DATA
        global CONTROL
        CONTROL['driveLeft'] = 0.5
        CONTROL['driveRight'] = 0.5
        while running:
            if self.autonomous:
                self.drive_autonoumous()
            else:
                self.drive_joystick()
                    
            # Change the LED to reflect the status of the EPO latch
            self.PBR.SetLed(self.PBR.GetEpo())
            # Wait for the interval period
            time.sleep(self.interval)
        # Disable all drives
        self.PBR.MotorsOff()


DATA = {}
CONTROL = {}
# Create the image buffer frame
lastFrame = None
lockFrame = threading.Lock()
running = True
imageWidth = 320                        # Width of the captured image in pixels
imageHeight = 160                       # Height of the captured image in pixels
frameRate = 4                           # Number of images to capture per second    

# Startup sequence
print('Setup camera')
camera = picamera.PiCamera()
camera.resolution = (imageWidth, imageHeight)
camera.framerate = frameRate
camera.hflip = True
camera.vflip = True

print('Setup the stream processing thread')
sys.stdout.flush()
processor = StreamProcessor(camera)

print('Wait ...')
time.sleep(2)
sys.stdout.flush()
captureThread = ImageCapture()