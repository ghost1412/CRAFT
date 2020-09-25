import numpy as np
import json
from lstm import lstm
from loaddata import load
from keras.models import *
import cv2

annotations, _ = load()
F=75
'''n_words=6414
n_tags=42
LSTM = lstm(hidden=64)
LSTM.load_weights('entity_lstm.h5')'''

from preprocess import preProcessData
from keras.models import Model, Input
from keras.layers import *
from keras_contrib.layers import CRF
from keras.utils import plot_model
from keras.models import load_model
import json
import numpy as np

from keras.callbacks import ModelCheckpoint
#from livelossplot import PlotLossesKeras


BATCH_SIZE = 512  # Number of examples used in each iteration
EPOCHS = 20  # Number of passes through entire dataset
MAX_LEN = 75  # Max length of review (in words)
EMBEDDING = 100  # Dimension of word embedding vector


def lstm(hidden=100, summary=True):
	n_words = 6414
	n_tags=42
	input = Input(shape=(MAX_LEN,))
	model = Embedding(input_dim=n_words+2, output_dim=EMBEDDING, # n_words + 2 (PAD & UNK)
		          input_length=MAX_LEN, mask_zero=True)(input)  # default: 20-dim embedding
	model = Bidirectional(LSTM(units=hidden, return_sequences=True, recurrent_dropout=0.1))(model)
	if hidden == 64:model = Bidirectional(LSTM(units=hidden, return_sequences=True, recurrent_dropout=0.1, return_state=True), merge_mode='concat')(model)
	else:model = Bidirectional(LSTM(units=hidden, return_sequences=True, recurrent_dropout=0.1, return_state=True), merge_mode='ave')(model)
	#model, forword_h, forword_c, backword_h, backword_c  = Bidirectional(LSTM(units=50, return_sequences=True, return_state=True,
		                   #recurrent_dropout=0.1))(model)  # variational biLSTM
	#state_h = Concatenate()([forword_h, backword_h])
	#state_c = Concatenate()([forword_c, backword_c])
	model = TimeDistributed(Dense(100, activation="relu", name='LSTM_Dense'))(model[0])  # a dense layer as suggested by neuralNer
	crf = CRF(n_tags+1)  # CRF layer, n_tags+1(PAD)
	out = crf(model)  # output
	model = Model(input, out)
	model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
	#plot_model(model, to_file='lstm.png', show_shapes=True)
	if summary:model.summary()

	return model


X_tr, X_te, y_tr, y_te, n_words, n_tags, idx2tag, words = preProcessData()
print("n_words:{}, n_tags:{}".format(n_words,n_tags))
LSTM = lstm()
history = LSTM.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)


with open('lstminput.json') as f:
	lstminput = json.load(f)

count=1
for anno in annotations:
	vid_name = anno['globalID']
	#if count < 54000:
	#	count += len(anno['objects'])
	#	count += len(anno['characters'])
	#	continue
	get_embeddings = K.function([LSTM.layers[0].input], [LSTM.layers[1].output[0]])
	embeddings = np.array(get_embeddings([np.array(np.reshape(np.array(lstminput[vid_name]), (1, 75,)))])[0])
	embeddings = np.reshape(embeddings, (75, 100))
	gt_video = np.load('Dataset/flintstones_dataset/video_frames/'+vid_name+'.npy')
	video = np.zeros((128,128,3*F), dtype=np.uint8)	
	entities = [] 
	for obj in anno['objects']:
		entities.append(obj)
	for car in anno['characters']:
		entities.append(car)
	sorted(entities, key=lambda x: x['entitySpan'][0])
	for entity in entities:
		print("[{}] ".format(count) + entity['globalID'] + " .... ")
		embedding = embeddings[(entity['entitySpan'][1] - 1),:]
		mask = np.load('Dataset/flintstones_dataset/entity_segmentation/'+entity['globalID']+'_segm.npy.npz')['arr_0']
		segmented = np.zeros((F,128,128,3), dtype=np.uint8)
		'''
		for i in range(3):
			segmented[:, :, :, i] = mask * gt_video[:, :, :, i]
		'''
		np.savez_compressed('Dataset/flintstones_dataset/layoutcomposer_in/'+entity['globalID'], Vi=video, embedding=embedding)
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
		#cv2.imwrite('../img_{}.jpg'.format(count), cv2.cvtColor(segmented[0],cv2.COLOR_RGB2BGR))
		#cv2.imwrite('../img_{}_vid.jpg'.format(count), cv2.cvtColor(video[:,:,0:3],cv2.COLOR_RGB2BGR))
		count = count + 1
