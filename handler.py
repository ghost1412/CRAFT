import numpy as np
from keras.utils import Sequence
import keras.backend as K
from keras.models import Sequential, Model

class DataGeneratorLayout(Sequence):

	def __init__(self, annotations, video_files, batch_size=32, F=75, LSTM=None, lstminput=None, graph=None):
		self.annotations = annotations
		self.video_files = video_files
		self.batch_size = batch_size
		self.F = F
		self.LSTM = LSTM
		self.lstminput = lstminput
		self.graph = graph

	def __len__(self):
		return int(np.ceil(len(self.video_files) / float(self.batch_size)))
	
	def __getitem__(self, idx):
		if (idx+1)*self.batch_size < len(self.video_files):
			batch = self.video_files[idx * self.batch_size:(idx + 1) * self.batch_size]
		else:
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
			#with self.graph.as_default():
				#get_embeddings = K.function([self.LSTM.layers[0].input], [self.LSTM.layers[4].output])
				#embeddings = np.array(get_embeddings([np.array(np.reshape(np.array(self.lstminput[vid]), (1, 75,)))])[0])

			#embeddings = np.reshape(embeddings, (75, 100))
			entities = []
			for obj in anno['objects']:
				entities.append(obj)
			for car in anno['characters']:
				entities.append(car)
			sorted(entities, key=lambda x: x['entitySpan'][0])
			for entity in entities:
				#batch_em.append(embeddings[int((entity['entitySpan'][0]+entity['entitySpan'][1])/2),:])
				inp = np.load('flintstones_dataset/layoutcomposer_in/'+entity['globalID']+'.npz')
				batch_vi.append(inp['Vi'][:,:,np.array([30,31,32,120,121,122,210,211,212])])
				batch_em.append(inp['embedding'])
				out = np.load('flintstones_dataset/layoutcomposer_gtF3/'+entity['globalID']+'.npz')
				Ploc.append(out['Ploc'])
				mu.append(out['mu'])
				#mask = np.load('flintstones_dataset/entity_segmentation/'+entity['globalID']+'_segm.npy.npz')['arr_0']
				#segmented = np.empty((self.F, 128,128,3), dtype=np.float64)
				#for i in range(3):
				#	segmented[:, :, :, i] = mask * gt_video[:, :, :, i]
				#frame = 0
				#for i in range(self.F):
				#	for l in range(3):
				#		for j in range(128):
				#			for k in range(128):
				#				if np.where(segmented[i][j][k][l] > 0):
				#					video[j][k][frame] = segmented[i][j][k][l]
				#	frame = frame + 1
				#batch_vi.append(video)
		return [np.array(batch_vi), np.array(batch_em)], [np.array(mu), np.array(Ploc)]


class DataGeneratorEntity(Sequence):

    def __init__(self, inputs, labels, batch_size):
        self.inputs, self.labels = inputs, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.inputs) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.inputs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([resize(imread(file_name), (200, 200)) for file_name in batch_x]), np.array(batch_y)



class DataGeneratorBackground(Sequence):

    def __init__(self, inputs, labels, batch_size):
        self.inputs, self.labels = inputs, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.inputs) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.inputs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([resize(imread(file_name), (200, 200)) for file_name in batch_x]), np.array(batch_y)


