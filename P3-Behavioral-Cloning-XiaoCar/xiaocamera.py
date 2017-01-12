import os
import pickle
import time
import io
import threading
import picamera
import picamera.array
import csv
import struct

class Camera(object):

    def __init__(self, DATA, CONTROL):
        self.data = DATA
        self.control = CONTROL

    def get_frame(self):
        with picamera.PiCamera() as camera:
            # camera setup
            camera.resolution = (320, 240)
            camera.hflip = True
            camera.vflip = True

            with picamera.array.PiYUVArray(camera) as stream:

                camera.resolution = (640, 480)
                camera.capture(stream, 'yuv')

                # store frame
                stream.seek(0)
                self.frame = stream.read()

                row = {}
                # grab the raw NumPy array representing the image, then initialize the timestamp
                # and occupied/unoccupied text
                dt = time.time()
                # row['yuv_image'] = stream.array
                row['rgb_image'] = stream.rgb_array
                row['time'] = dt
                row['left'] = self.control['driveLeft']
                row['right'] = self.control['driveRight']
                self.data[str(dt)] = row
                # reset stream for next frame
                stream.seek(0)
                stream.truncate()
                print('caputred ', self.control['driveLeft'], self.control['driveRight'])
                print()
