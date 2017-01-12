import smbus
import struct
import sys
sys.path.append('..')

from xiaocamera import Camera

class Driver(object):

    def __init__(self, app):
        self.bus = smbus.SMBus(1)

    def _write_bus(self, address, args):
        self.bus.write_i2c_block_data(20, address, args)

    def drive(self, left, right):
        args = list(struct.pack('hh', *[left, right]))
        self._write_bus(6, args)