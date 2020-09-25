import numpy as np
import json
from lstm import lstm
from loaddata import load
from keras.models import *
import cv2

annotations, _ = load()
F=75
n_words=6414
n_tags=42
LSTM = lstm()
LSTM.load_weights('lstm.h5')

with open('lstminput.json') as f:
	lstminput = json.load(f)

count=1
for anno in annotations:
	vid_name = anno['globalID']
	get_embeddings = K.function([LSTM.layers[0].input], [LSTM.layers[4].output])
	embeddings = np.array(get_embeddings([np.array(np.reshape(np.array(lstminput[vid_name]), (1, 75,)))])[0])
	embeddings = np.reshape(embeddings, (75, 100))
	gt_video = np.load('flintstones_dataset/video_frames/'+vid_name+'.npy')
	video = np.zeros((128,128,3*F), dtype=np.uint8)	
	entities = [] 
	for obj in anno['objects']:
		entities.append(obj)
	for car in anno['characters']:
		entities.append(car)
	sorted(entities, key=lambda x: x['entitySpan'][0])
	for entity in entities:
		print("[{}] ".format(count) + entity['globalID'] + " .... ", end='')
		embedding = embeddings[int((entity['entitySpan'][0]+entity['entitySpan'][1])/2),:]
		mask = np.load('flintstones_dataset/entity_segmentation/'+entity['globalID']+'_segm.npy.npz')['arr_0']
		segmented = np.zeros((F,128,128,3), dtype=np.uint8)
		'''
		for i in range(3):
			segmented[:, :, :, i] = mask * gt_video[:, :, :, i]
		'''
		np.savez_compressed('flintstones_dataset/layoutcomposer_in/'+entity['globalID'], Vi=video, embedding=embedding)
		frame = 0
		for i in range(F):
			vidframe = gt_video[i]
			segmented[i] = cv2.bitwise_and(vidframe, vidframe, mask=mask[i])
			'''
			for l in range(3):
				bmask = np.bool_(mask[i])
				video[bmask,frame] = segmented[i,bmask,l]
				frame = frame + 1
			'''
			bmask = np.bool_(mask[i])
			video[bmask,i*3:(i+1)*3] = segmented[i,bmask]
		print("Done")
		#cv2.imwrite('../img_{}.jpg'.format(count), cv2.cvtColor(segmented[0],cv2.COLOR_RGB2BGR))
		#cv2.imwrite('../img_{}_vid.jpg'.format(count), cv2.cvtColor(video[:,:,0:3],cv2.COLOR_RGB2BGR))
		count = count + 1
