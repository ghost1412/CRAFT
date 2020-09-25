import numpy as np
import json
from loaddata import load

annotations,_ = load()
count = 0
F = 8
for anno in annotations:
	entities = []
	for obj in anno['objects']:
		entities.append(obj)
	for car in anno['characters']:
		entities.append(car)
	for entity in entities:
		print('[{}] '.format(count)+entity['globalID']+' .... ', end = '')
		Ploc = np.zeros((128,128,F), dtype=np.float64)
		mu = np.zeros((2*F,), dtype=np.float64)
		rect = np.load('Dataset/flintstones_dataset/entity_tracking/'+entity['globalID']+'.npy')
		for j in range(F):
			i = int(j*75/8)
			if len(rect[i]) == 0:
				continue
			dx = abs(rect[i][2] - rect[i][0])
			dy = abs(rect[i][3] - rect[i][1])
            
			sx = int((rect[i][2] + rect[i][0])/2 - dx//2)
			sy = int((rect[i][3] + rect[i][1])/2 - dy//2)
			if sx > 127:
				sx = 127
			if sy > 127:
				sy = 127
			if sx < 0:
				sx = 0
			if sy < 0:
				sy = 0
			sx_L = max(sx - 2, 0)
			sx_R = min(sx + 2, 127)
			sy_U = max(sy - 2, 0)
			sy_D = min(sy + 2, 127)
			Ploc[sx_L:sx_R, sy_U:sy_D, j] = 1
			mu[2*j] = dx/128
			mu[2*j+1] = dy/128
		np.savez_compressed('Dataset/flintstones_dataset/layoutcomposer_gtF8/'+entity['globalID'], Ploc=Ploc, mu=mu)
		print('done')
		count = count + 1
