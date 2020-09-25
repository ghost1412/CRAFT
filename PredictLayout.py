from LayoutComposer import LayoutComposer
import cv2 as cv2
import Keras.backend as K
from keras.models import *
import json
from loaddata import load
with open('layoutannotations.json') as f:
	annotations = json.load(f)
_, split = load()

layoutcomposer = load_model('LayoutComposerModel')

batch = self.video_files[idx * self.batch_size:]
batch_vi = []
batch_em = []
Ploc = []
mu = []
K.set_floatx('float32')
for i in range(len(batch)):
	vid = batch[i]
	gt_video = np.load('flintstones_dataset/video_frames/'+vid+'.npy')
	video = np.empty((128,128,3*self.F), dtype=np.float64)
	anno = self.annotations[vid]
	entities = []
	for obj in anno['objects']:
		entities.append(obj)
	for car in anno['characters']:
		entities.append(car)
	sorted(entities, key=lambda x: x['entitySpan'][0])
	for entity in entities:
		inp = np.load('flintstones_dataset/layoutcomposer_in/'+entity['globalID']+'.npz')
		batch_vi.append(inp['Vi'][:,:,np.array([30,31,32,120,121,122,210,211,212])])
		batch_em.append(inp['embedding'])
		out = np.load('flintstones_dataset/layoutcomposer_gtF3/'+entity['globalID']+'.npz')
		Ploc.append(out['Ploc'])
		mu.append(out['mu'])
prediction = layoutcomposer.predict(np.array([batch_vi, batch_em]))
Ploc_pred = predition[0]
mu_pred = predition[1]

