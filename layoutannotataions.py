import json
import numpy as np
from loaddata import load

def fuck():
	annotations, split = load()
	layoutanno = dict()
	i = 1
	for annotation in annotations:
		layoutanno.update({annotation['globalID']:annotation})
		print("[{}] ".format(i)+annotation['globalID'])
		i = i + 1
	with open('layoutannotations.json','w') as fp:
		json.dump(layoutanno,fp)
	return

fuck()

