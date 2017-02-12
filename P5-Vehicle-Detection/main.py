import os
import pickle

import argparse
import cv2

import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip

from xiaodetector import Tracker, Classifier

def make_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--model', dest='model', help='Path to pre-trained model')
	parser.add_argument('--create-model', dest='create_model')

	return parser

def return_model(args):
	load = True
	if args.create_model:
		load = False

	if load and args.model and os.path.exists(args.model):
		print("loading stored model")
		return pickle.load(open('./xiaodetector/model/model.p', 'rb'))

	if not args.model or not os.path.exists('./xiaodetector/model/model.p'):
		svm = Classifier()
		
		svm.load_data()
		
		svm.save_model()
		
		return pickle.load(open('./model/model.p', 'rb'))

if __name__ == '__main__':

	tracker = Tracker()

	white_output = 'output_video.mp4'

	clip1 = VideoFileClip("project_video.mp4")

	# NOTE: this function expects color images!!
	white_clip = clip1.fl_image(tracker.track)

	white_clip.write_videofile(white_output, audio=False)

	# for img in os.listdir('./test_images'):
	# 	try:
	# 		if not '.jpg' in img: continue
	# 		image = cv2.imread('./test_images/'+img)
	# 		im = tracker.track(image)
	# 		cv2.imwrite('./output_images/'+img, im)
	# 		# break
	# 	except Exception as e:
	# 		raise e
	# 		break
	# svm = Classifier()
	
	# svm.load_data()

	# svm.save_model()

	# svm.score()		
	
