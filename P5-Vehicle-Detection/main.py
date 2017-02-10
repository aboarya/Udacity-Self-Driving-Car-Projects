import argparse

from xiaodetector import Classifier

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
		return pickle.load(open('./model/model.p', 'rb'))

	if not args.model or not os.path.exists('./model/model.p'):
		svm = Classifier()
		
		svm.load_data()
		
		svm.save_model()
		
		return pickle.load(open('./model/model.p', 'rb'))

if __name__ == '__main__':
	parser = make_args()

	args = parser.parse_args()

	model = return_model(args)

