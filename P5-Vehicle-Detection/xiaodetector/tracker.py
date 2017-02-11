import collections

from .detector import VehicleDetector as Detector

class VehicleTracker(object):

	def __init__(self):
		self.detector = Detector()

	def track(self, image):
		im, labels = self.detector.detect(image)
		
		return im