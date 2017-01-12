#!/usr/bin/env python3
# coding: Latin-1

# Load library functions we want
import time
import os
import pickle
import sys
import threading
import picamera
sys.path.append('..')

import pygame
import cv2
from apscheduler.schedulers.background import BackgroundScheduler

import PicoBorgRev
from xiaocamera import Camera

sys.stdout = sys.stderr

imageWidth = 240                        # Width of the captured image in pixels
imageHeight = 180                       # Height of the captured image in pixels
frameRate = 2                           # Number of images to capture per second
displayRate = 2                         # Number of images to request per second

# Global values
global lastFrame
global lockFrame
global camera
global processor
global running
global watchdog
running = True

# Create the image buffer frame
lastFrame = None
lockFrame = threading.Lock()

# Timeout thread
class Watchdog(threading.Thread):
    def __init__(self):
        super(Watchdog, self).__init__()
        self.event = threading.Event()
        self.terminated = False
        self.start()
        self.timestamp = time.time()

    def run(self):
        timedOut = True
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for a network event to be flagged for up to one second
            if timedOut:
                if self.event.wait(1):
                    # Connection
                    print('Reconnected...')
                    timedOut = False
                    self.event.clear()
            else:
                if self.event.wait(1):
                    self.event.clear()
                else:
                    # Timed out
                    print('Timed out...')
                    timedOut = True
                    PBR.MotorsOff()

# Image stream processing thread
class StreamProcessor(threading.Thread):
    def __init__(self):
        super(StreamProcessor, self).__init__()
        self.stream = picamera.array.PiRGBArray(camera)
        self.event = threading.Event()
        self.terminated = False
        self.start()
        self.begin = 0

    def run(self):
        global lastFrame
        global lockFrame
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    # Read the image and save globally
                    self.stream.seek(0)
                    flippedArray = cv2.flip(self.stream.array, -1) # Flips X and Y
                    retval, thisFrame = cv2.imencode('.jpg', flippedArray)
                    del flippedArray
                    lockFrame.acquire()
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
        global running
        while running:
            if processor.event.is_set():
                time.sleep(0.01)
            else:
                yield processor.stream
                processor.event.set()

def save_data(DATA):

    if os.path.exists('robot-S-track.p'):
        with open('robot-S-track.p', 'rb') as _file:
            data = pickle.load(_file)
    else:
        data = {}
    DATA.update(data)
    with open('robot-S-track.p', 'wb') as _file:
        pickle.dump(DATA, _file)   
    return ""

def setup_joystick():

    # Setup pygame
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Removes the need to have a GUI window
    pygame.init()
    pygame.joystick.init()
    pygame.display.set_mode((1,1))
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick

def setup_driver():

    # Setup the PicoBorg Reverse
    PBR = PicoBorgRev.PicoBorgRev()
    PBR.Init()
    if not PBR.foundChip:
        boards = PicoBorgRev.ScanForPicoBorgReverse()
        if len(boards) == 0:
            print('No PicoBorg Reverse found, check you are attached :)')
        else:
            print('No PicoBorg Reverse at address %02X, but we did find boards:' % (PBR.i2cAddress))
            for board in boards:
                print('    %02X (%d)' % (board, board))
            print('If you need to change the I²C address change the setup line so it is correct, e.g.')
            print('PBR.i2cAddress = 0x%02X' % (boards[0]))
        sys.exit()

    PBR.ResetEpo()

    return PBR


def main():
    try:
        DATA = {}
        CONTROL = {}
        # Settings for the joystick
        axisUpDown = 1                          # Joystick axis to read for up / down position
        axisUpDownInverted = False              # Set this to True if up and down appear to be swapped
        axisLeftRight = 2                       # Joystick axis to read for left / right position
        axisLeftRightInverted = False           # Set this to True if left and right appear to be swapped
        buttonResetEpo = 3                      # Joystick button number to perform an EPO reset (Start)
        buttonSlow = 8                          # Joystick button number for driving slowly whilst held (L2)
        slowFactor = 0.5                        # Speed to slow to when the drive slowly button is held, e.g. 0.5 would be half speed
        buttonFastTurn = 9                      # Joystick button number for turning fast (R2)
        interval = 0.00                         # Time between updates in seconds, smaller responds faster but uses more processor time
        running = True
        hadEvent = False
        upDown = 0.0
        leftRight = 0.0
        CONTROL['driveLeft'] = 0.0
        CONTROL['driveRight'] = 0.0
        # Setup the Camera
        camera = Camera(DATA, CONTROL)
        joystick = setup_joystick()
        PBR = setup_driver()
        sched = BackgroundScheduler()
        sched.add_job(camera.get_frame, 'interval', seconds=1)

        print('Press CTRL+C to quit')
        
        # Loop indefinitely
        while running:
            # Get the latest events from the system
            hadEvent = False
            events = pygame.event.get()
            # Handle each event individually
            for event in events:
                if event.type == pygame.QUIT:
                    # User exit
                    running = False
                elif event.type == pygame.JOYBUTTONDOWN:
                    # A button on the joystick just got pushed down
                    hadEvent = True                    
                elif event.type == pygame.JOYAXISMOTION:
                    # A joystick has been moved
                    hadEvent = True
                if hadEvent:
                    # Read axis positions (-1 to +1)
                    if axisUpDownInverted:
                        upDown = -joystick.get_axis(axisUpDown)
                    else:
                        upDown = joystick.get_axis(axisUpDown)
                    if axisLeftRightInverted:
                        leftRight = -joystick.get_axis(axisLeftRight)
                    else:
                        leftRight = joystick.get_axis(axisLeftRight)
                    # Apply steering speeds
                    if not joystick.get_button(buttonFastTurn):
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
                    # Check for button presses
                    if joystick.get_button(buttonResetEpo):
                        PBR.ResetEpo()
                    if joystick.get_button(buttonSlow):
                        CONTROL['driveLeft'] *= slowFactor
                        CONTROL['driveRight'] *= slowFactor
                    # Set the motors to the new speeds
                    PBR.SetMotor1(CONTROL['driveLeft'])
                    PBR.SetMotor2(CONTROL['driveRight'])
                    # Camera snapshot
                    lockFrame.acquire()
                    sendFrame = lastFrame
                    lockFrame.release()
                    if sendFrame != None:
                        dt = time.time()
                        DATA[str(dt)] = {'img' : sendFrame.tostring(), 'left': CONTROL['driveLeft'], 'right': CONTROL['driveRight']}
                    
            # Change the LED to reflect the status of the EPO latch
            PBR.SetLed(PBR.GetEpo())
            # Wait for the interval period
            time.sleep(interval)
        # Disable all drives
        PBR.MotorsOff()
    except KeyboardInterrupt:
        # CTRL+C exit, disable all drives
        PBR.MotorsOff()
        save_data(DATA)

    print()

# Startup sequence
print('Setup camera')
camera = picamera.PiCamera()
camera.resolution = (imageWidth, imageHeight)
camera.framerate = frameRate
camera.hflip = True
camera.vflip = True

print('Setup the stream processing thread')
processor = StreamProcessor()

print('Wait ...')
time.sleep(2)
captureThread = ImageCapture()

print('Setup the watchdog')
watchdog = Watchdog()

main()