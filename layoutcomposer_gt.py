import numpy as np
import json
from loaddata import load

annotations,_ = load()
count = 0
for anno in annotations:
	entities = []
	for obj in anno['objects']:
		entities.append(obj)
	for car in anno['characters']:
		entities.append(car)
	for entity in entities:
		print('[{}] '.format(count)+entity['globalID']+' .... ', end = '')
		Ploc = np.zeros((128,128,3), dtype=np.float64)
		mu = np.zeros((6,), dtype=np.float64)
		for i in range(3):
			if len(entity['rectangles'][i]) == 0:
				continue
			sx = int((entity['rectangles'][i][2] + entity['rectangles'][i][0])/2)
			sy = int((entity['rectangles'][i][3] + entity['rectangles'][i][1])/2)
			if sx > 127:
				sx = 127
			if sy > 127:
				sy = 127
			if sx < 0:
				sx = 0
			if sy < 0:
				sy = 0
			dx = abs(entity['rectangles'][i][2] - entity['rectangles'][i][0])
			dy = abs(entity['rectangles'][i][3] - entity['rectangles'][i][1])
			Ploc[sx][sy][i] = 1
			mu[2*i] = dx/128
			mu[2*i+1] = dy/128
		np.savez_compressed('flintstones_dataset/layoutcomposer_gtF3/'+entity['globalID'], Ploc=Ploc, mu=mu)
		print('done')
		count = count + 1
