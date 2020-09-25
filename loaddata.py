import numpy as np
import json
from pprint import pprint


def load():
	#print("Loading annotations .... ", end='')
	with open('flintstones_annotations_v1-0.json') as annotations:
		flintstones_annotations = json.load(annotations)
	print("done")

	#print("Loading test train val split files .... ", end='')
	with open('train-val-test_split.json') as split:
		train_val_test_split = json.load(split)
	print("done")
	return flintstones_annotations, train_val_test_split

def trainData(train_val_test_split):
	train = []
	for file in train_val_test_split['train']:
		video = np.load('flintstones_dataset/video_frames/'+file+'.npy')
		train.append(video)
	return np.array(train)

def valData(train_val_test_split):
	val = []
	for file in train_val_test_split['val']:
		video = np.load('flintstones_dataset/video_frames/'+file+'.npy')
		val.append(video)
	return np.array(val)

def testData(train_val_test_split):
	test = []
	for file in train_val_test_split['test']:
		video = np.load('flintstones_dataset/video_frames/'+file+'.npy')
		test.append(video)
	return np.array(test)

def getVideo(name):
	return np.load('flintstones_dataset/video_frames/'+name+'.npy')


