import numpy as np
import glob
import os

def groundTruth(file, F):
	rects = np.load('flintstones_dataset/entity_tracking/'+file)
	mu = np.zeros(2*F)
	Ploc = np.zeros((128,128,F))
	i = 0
	for rect in rects:
		cx = int((rect[0]+rect[2])//2)
		cy = int((rect[1]+rect[3])//2)
		if cx > 127:
			cx = 127
		if cy > 127:
			cy = 127
		if cx < 0:
			cx = 0
		if cy < 0:
			cy = 0
		Ploc[cx,cy,i] = 1
		mu[2*i] = abs(rect[0]-rect[2])/128
		mu[2*i+1] = abs(rect[1]-rect[3])/128
		i = i + 1
	return Ploc, mu


data_files = glob.glob('flintstones_dataset/entity_tracking/*')
i = 1
for file in data_files:
	file = os.path.basename(file)
	print('[{}] '.format(i)+file+' .... ',end="")
	Ploc, mu = groundTruth(file, 75)
	temp = {'Ploc':Ploc, 'mu':mu}
	np.savez_compressed('flintstones_dataset/layoutcomposer_gt/'+file, Ploc=Ploc, mu=mu)
	print('done')
	i = i + 1
