from keras.models import Sequential, Model
from keras.layers import *
from keras.models import *
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from attention_layer import AttentionLayer
from time import time
from loaddata import load
from handler import DataGeneratorLayout
from keras.models import load_model
from lstm import lstm
import tensorflow as tf
import json
from keras import backend as K
import math as m
from keras.losses import mse, binary_crossentropy
import numpy as np
import os
from keras.regularizers import l2
from keras.layers import LeakyReLU, Softmax
from keras.backend import slice
from numpy import asarray
from numpy import zeros
from numpy import unravel_index


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#from livelossplot import PlotLossesKeras
'''
def bilinear_kernel(h, w, channels, use_bias = True, dtype = "float32") :

	y = np.zeros((h,w,channels,channels), dtype = dtype)
	for i in range(0, h):
		for j in range(0, w):
			y[i,j,:,:] = np.identity(channels) / float(h*w*1)
	if use_bias : return [y,np.array([0.], dtype = dtype)]
	else : return [y]
'''
class LayoutComposer():

	def __init__(self,video_width=128, video_height=128, gt_directory=None, video_directory=None, annotations=None, F=75, LSTM=None,graph=None):
		if gt_directory is None:
			gt_directory = 'flintstones_dataset/layoutcomposer_gtF3/'
		if video_directory is None:
			video_directory = 'flintstones_dataset/video_frames/'
		if annotations is None:
			annotations = 'layoutinput.json'
	
		self.video_width = video_width
		self.video_height = video_height
		self.gt_directory = gt_directory
		self.video_directory = video_directory
		self.F = F
		self.annotations = annotations
		self.LSTM = LSTM
		self.model = None
		self.graph=graph
		self.history = None
	
	
	def channelPool(self,x):
		return K.max(x,axis=-1)
	
	def channelAvg(self,x):
		return K.mean(x,axis=-1)
    
	def global_average_pooling(self, x):
		return K.mean(x, axis = -1)
    
	def global_average_pooling_shape(self, input_shape):
		return input_shape[0:2]
    
	def softmax2d(self, x):
		Exp = K.exp(x)
		return Exp / K.sum(Exp, axis=(1,2), keepdims=True)	
		#label_dim = -1
		#d = K.exp(x - K.max(x, axis=label_dim, keepdims=True))
		#return d / K.sum(d, axis=label_dim, keepdims=True)

	def categorical_crossentropy2d(self, y_true, y_pred):
		__EPS = 1e-5
		y_pred = K.clip(y_pred, __EPS, 1 - __EPS)
		#x0, y0 = unravel_index(Ploc.argmax(axis=(1,2)), Ploc[:,:,:,1].shape)
		#return (K.log(y_pred[:,x0,y0]))
		#return -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
		return -K.mean(y_true*K.log(y_pred))

	def lossfn(self,y_true, y_pred):
		K.set_floatx('float32')
		Ploc_true = (y_true[0])
		mu_true = (y_true[1])
		Ploc_pred = (y_pred[0])
		mu_pred = (y_pred[1])
		print(y_pred[1])
		sigma = K.cast_to_floatx(np.diag(np.full((2*self.F,), 0.005)))
		cost = (-tf.log(Ploc_pred) + 
				(0.5) * tf.log(tf.matrix_determinant(sigma)) + 
				(0.5) * K.dot(tf.transpose(mu_true - mu_pred), K.dot(tf.matrix_inverse(sigma), (mu_true - mu_pred))) + 
				tf.log(2 * m.pi))
		return cost
	
	def loss1(self, y_true, y_pred):
		#return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
		#return tf.log(y_true-y_pred)
		#return binary_crossentropy(y_true, y_pred)
		#likelihood = K.tf.distributions.Bernoulli(logits=y_pred)
		#return  -tf.reduce_sum(log(y_pred))
		return K.sum(K.sum(K.sum(-y_true * K.log(y_pred), axis=1)))/(self.F * tf.cast(tf.shape(y_true)[0], tf.float32))
		#dist = K.tf.distributions.Normal(loc=y_pred, scale=0.005)
		#return -1.0 * dist.log_prob(y_true)

	def loss2(self, y_true, y_pred):
		sigma = K.cast_to_floatx(np.diag(np.full((2*self.F,), 0.005)))
		#loss = (0.5) * K.dot(K.dot((y_true- y_pred), tf.matrix_inverse(sigma)), tf.transpose(y_true- y_pred)) + self.F * tf.log(2*m.pi) - 0.5*tf.log(tf.linalg.det(sigma))
		#return loss / (self.F * tf.cast(tf.shape(y_true)[0], tf.float32))
		#9.2041199827
		#batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
		det = (0.5) * self.F * tf.log(tf.matrix_determinant(sigma))
		log2pi = self.F * tf.log(2 * np.pi)
		dot = (0.5) * K.sum(K.dot((y_true - y_pred), tf.matrix_inverse(sigma)) * (y_true - y_pred), axis=-1)
		return K.mean(dot)
		#n_dims = 2*self.F
		#mu = tf.cast(y_pred, tf.float32)
		#y_true = tf.cast(y_true, tf.float32)

		#logsigma = tf.cast(np.full((1, 2*self.F), 0.005), tf.float32)

		#mse = 0.5*K.sum(K.square((y_true-mu)/K.exp(logsigma)),axis=1)
		#sigma_trace = K.sum(logsigma, axis=1)
		#log2pi = 0.5*n_dims*np.log(2*np.pi)

		#log_likelihood = mse+sigma_trace+log2pi

		#return K.mean(log_likelihood)

	def createModel(self,summary=True):
		Vi_1 = Input((self.video_height, self.video_width, self.F*3), name='Vi_1')
		
		CNN1 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(Vi_1)
		CNN1 = LeakyReLU(alpha=0.3)(CNN1)
		CNN1 = BatchNormalization()(CNN1)
		CNN2 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(CNN1)
		CNN2 = LeakyReLU(alpha=0.3)(CNN2)
		CNN2 = BatchNormalization()(CNN2)
		CNN3 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(CNN2)
		CNN3 = LeakyReLU(alpha=0.3)(CNN3)
		CNN3 = BatchNormalization()(CNN3)
		CNN = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(CNN3)
		CNN = LeakyReLU(alpha=0.3)(CNN) 
		CNN = BatchNormalization()(CNN)
		#Ei = Input((100,), name='Ei')
		Ei = Input(shape=(75, ))
		Ei1 = Embedding(input_dim=6414+2, output_dim=100, # n_words + 2 (PAD & UNK)
		          input_length=75, mask_zero=True)(Ei)
		BiLstm = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(Ei1)
		BiLstm1 = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1, return_state=True), merge_mode='ave')(BiLstm)
		'''embeddings_index = dict()
		f = open('Dataset/glove.840B.300d.txt')
		for line in f:
			values =  line.split(' ')
			word = values[0]
			coefs = np.array(values[1:],dtype = 'float32')
			embeddings_index[word] = coefs
		f.close()
		embedding_matrix = zeros((6414+2, 100))
		for word, i in t.word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
		Em = Embedding(6414+2, 100, weights=[embedding_matrix], input_length=75, trainable=False)(Ei)
		# CONCATINATE WITH Embeddings
		#print(BiLstm1[0][:, 0, :])'''
		En = Input(shape=(2, ), dtype='int64')
		#EnEm = Lambda(lambda x: slice(x, 0, 1))(BiLstm1[0])
		
		def getEmbedding(x):
		#	print('fdf')
			embed = x[0]
			sess = K.get_session()
			entity = sess.run(x[1])
			#with sess.as_default():
			entity = K.eval(entity)
			entity = K.get_value(entity)
			return embed[:, entity, :]

		def getEmb(x):
			embed = x[0]
			en = x[1]
			#return K.gather(embed, en)
			en = tf.dtypes.cast(en, dtype=tf.int32)
			return tf.gather_nd(embed, en)
		#getEmbedding = Lambda(lambda embed, entity: embed[:, entity, :], output_shape=(100,))
		#getEmbedding.arguments = {'entity': En[0]}
		EnEm = Lambda(getEmb)([BiLstm1[0], En])
		EnD = Dense(100, kernel_regularizer=l2(0.0001))(EnEm)
		EnD = LeakyReLU(alpha=0.3)(EnD)	
		#EnEm = getEmbedding(BiLstm1[0])
		repli = RepeatVector(32)(EnD)
		repli = Reshape((32*100,))(repli)
		repli = RepeatVector(32)(repli)
		repli = Reshape((32,32,100))(repli)	
	
		merged = concatenate([CNN, repli])

		#2D Grid coordinates

		#G2D = Input((32,32,2), name='G2D')
		#merged = concatenate([merged, G2D])

		# FULLY CONV. LOCATION MLP

		Ploc1 = Conv2D(256, kernel_size=(1, 1), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(merged) 
		Ploc1 = LeakyReLU(alpha=0.3)(Ploc1)
		Ploc1 = BatchNormalization()(Ploc1)
		Ploc2 = Conv2D(128, kernel_size=(1, 1), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(Ploc1)
		Ploc2 = LeakyReLU(alpha=0.3)(Ploc2) 
		Ploc2 = BatchNormalization()(Ploc2)
		#Ploc3 = Conv2D(128, kernel_size=(1, 1), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(Ploc2)
		#Ploc3 = LeakyReLU(alpha=0.3)(Ploc3) 
		#Ploc3 = BatchNormalization()(Ploc3)
		Ploc4 = Conv2D(self.F, kernel_size=(1, 1), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(Ploc2)
		Ploc4 = LeakyReLU(alpha=0.3)(Ploc4) 	
		Ploc4 = BatchNormalization()(Ploc4)
		# UPSAMPLING FOR LOCATION OUTPUT
		Ploc = UpSampling2D(size=(4, 4), data_format=None, interpolation='bilinear')(Ploc4)
		#Ploc = Reshape((128*128, 8))(Ploc)
		Ploc = Activation(self.softmax2d)(Ploc)
		#Ploc = Softmax(axis=1)(Ploc)
		def repeat2D(x):
			return K.repeat_elements(x, 612, 3)

		# CHANNEL MAXPOOLING AND MERGE WITH CNN
		Max = Lambda(self.channelPool)(Ploc4)
		AttentionMap = Reshape((32, 32, 1))(Max)
		AttentionMap = Lambda(repeat2D)(AttentionMap)
		attention = multiply([merged, AttentionMap])
		#AttentionMap = merge([merged, AttentionMap],  mode='mul')  
		attention = GlobalAveragePooling2D(data_format='channels_last')(attention)
		#print(attention)

		#attention = Lambda(self.channelAvg)(attention)
		#attention = AttentionLayer()([Avg, Max])
		#print(attention)
		#attention = Flatten()(attention)
		#attention = Reshape((32*32,))(attention)

		# SCALE MLP
		mu1 = Dense(256, kernel_regularizer=l2(0.0001))(attention)
		mu1 = LeakyReLU(alpha=0.3)(mu1)
		mu2 = Dense(128, kernel_regularizer=l2(0.0001))(mu1)
		mu2 = LeakyReLU(alpha=0.3)(mu2)
		mu = Dense(2*self.F, activation='sigmoid')(mu2)
		#mu = LeakyReLU(alpha=0.3)(mu) 
		# MODEL SUMMARY
		self.model = Model(inputs=[Vi_1, Ei, En], outputs=[mu, Ploc])
		if summary:self.model.summary()
		#plot_model(self.model, to_file='layoutcomposer_model.png', show_shapes=True)
		
		# COMPILE MODEL
		opt = Adam(lr=0.1, decay=0.5, amsgrad=False)	#weight decay 0.0001
		self.model.compile(optimizer=opt, loss={'dense_4':self.loss2, 'activation_1':self.categorical_crossentropy2d}, loss_weights=[1, 1], metrics=[])
		#self.model.compile(optimizer=opt, loss={'dense_3':self.loss2, 'activation_1':self.loss1}, loss_weights=[1, 1], metrics=[])
		return self.model


	def fit(self, epochs, batch_size=32, verbose=1, workers=30):

		self.createModel()
		#self.model.load_weights('LayoutCheckpoints/weights-improvement-15-15603.58.hdf5')
                #opt = Adam(lr=0.001, decay=0.5, amsgrad=False)  #weight decay 0.0001
                #self.model.compile(optimizer=opt, loss={'dense_4':self.loss2, 'activation_1':self.loss1}, loss_weights=[1, 1], metrics=[])
		with open('layoutannotations.json') as f:
			annotations = json.load(f)
		_, split = load()
		with open('lstminput.json') as f:
			lstminput = json.load(f)
		tf.flags.DEFINE_integer("batch_size", 32, "Batch size during training")
		tf.flags.DEFINE_integer("eval_batch_size", 8, "Batch size during evaluation")
		
		filepath="LayoutCheckpoints/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

		training_generator = DataGeneratorLayout(annotations=annotations, video_files=split['train'], F=self.F, batch_size=batch_size, LSTM=self.LSTM, lstminput=lstminput, graph=self.graph)
		validation_generator = DataGeneratorLayout(annotations=annotations, video_files=split['val'], F=self.F, batch_size=batch_size, LSTM=self.LSTM, lstminput=lstminput, graph=self.graph)
		self.history = self.model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, use_multiprocessing=True, workers=workers, verbose=verbose, callbacks=[checkpoint])
		self.model.save('LayoutComposerModel.h5')			
		'''			
		print("Training...")
		print("===============================================================================")
		bar = progressbar.ProgressBar(maxval=batch_count, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		for epoch in range(epochs):
			print("Epoch: {}/{}".format(epoch+1,epochs)
			epoch_start = time()
			bar.start()
			for batch in range(batch_count):
				batch_video_filesnames = video_filenames[batch*batch_size:(batch+1)*batch_size]
				batch_descriptions = descriptions[batch*batch_size:(batch+1)*batch_size]
				for vid in range(batch_size):
					video = np.load(self.video_directory + video_filenames[vid] + '.npy')
					Ploc
					for entity in range(entites):
						
					
				bar.update(batch+1)
			bar.finish()
		print("Time: %s secs " % (time() - epoch_start()), end='|')
		print(" train_loss: {}, val_loss: {}".format(train_loss, val_loss))
		'''
	def save_history(self):
		with open('LayoutTrainHistory.json', 'w') as fp:
			json.dump(self.history.history, fp)

#

layoutcomposer = LayoutComposer(F=8)
#layoutcomposer.model.load_weights('LayoutCheckpoints/weights-improvement-15-15603.58.hdf5')
layoutcomposer.fit(300, 32)

# layoutcomposer = LayoutComposer(F=8)
# layoutcomposer.model.load_weights('LayoutCheckpoints/weights-improvement-15-15603.58.hdf5')
# layoutcomposer.fit(200, 32)
# layoutcomposer.save_history()

'''
# +
from keras.models import load_model
#from LayoutComposer import LayoutComposer
import cv2
import numpy as np
import glob
from numpy import unravel_index
import matplotlib.pyplot as plt
from loaddata import load
from pprint import pprint


model = layoutcomposer.model

import sys
np.set_printoptions(threshold=sys.maxsize)
Entities = []
Embeddings = []
Trackings = []
gridCoordinate = []
global_ID = 's_06_e_25_shot_026136_026210'
video = np.load('Dataset/flintstones_dataset/video_frames/'+global_ID+'.npy')
file_names = glob.glob('Dataset/flintstones_dataset/layoutcomposer_in/'+global_ID+'*')

grid = np.empty((32, 32, 2))
for i in range(32):
	for j in range(32):
		grid[i][j][1] = i
		grid[i][j][0] = j	

for file_name in file_names:
    data_point = np.load(file_name)
    Entities.append(data_point['Vi'][:,:,np.array([0,1,2,15,16,17,30,31,32,72,73,74,120,121,122,162,163,164,183,184,185,210,211,212])])
    Embeddings.append(data_point['embedding'])
    gridCoordinate.append(grid)
    #print(Embeddings)
    #plt.imshow(data_point['Vi'][:, :, 0], cmap = 'gray', interpolation = 'bicubic')
    #plt.show()
file_names = glob.glob('Dataset/flintstones_dataset/entity_tracking/'+global_ID+'*')
for file_name in file_names:
    Trackings.append(np.load(file_name))
    print(file_name)
    print(np.load(file_name))


#data_point = np.load('flintstones_dataset/layoutcomposer_in/s_06_e_25_shot_033757_033831_char_0.npz')
#Entities.append(data_point['Vi'][:,:,np.array([30,31,32,120,121,122,210,211,212])])
#Embeddings.append(data_point['embedding'])
#data_point = np.load('flintstones_dataset/layoutcomposer_in/s_06_e_25_shot_033757_033831_char_1.npz')
#Entities.append(data_point['Vi'][:,:,np.array([30,31,32,120,121,122,210,211,212])])
#Embeddings.append(data_point['embedding'])
#data_point = np.load('flintstones_dataset/layoutcomposer_in/s_03_e_16_shot_028039_028113_char_0.npz')
#Entities.append(data_point['Vi'][:,:,np.array([30,31,32,120,121,122,210,211,212])])
#Embeddings.append(data_point['embedding'])
#data_point = np.load('flintstones_dataset/layoutcomposer_in/s_06_e_25_shot_016951_017025_char_0.npz')
#Entities.append(data_point['Vi'][:,:,np.array([30,31,32,120,121,122,210,211,212])])
#Embeddings.append(data_point['embedding'])

prediction = model.predict([Entities, gridCoordinate, Embeddings])
mus = prediction[0]
Plocs = prediction[1]
#pprint(np.sum(np.sum(Plocs[1, :, :, 1])))
#Plocs = np.reshape(Plocs, (np.array(mus).shape[0], 128,128,8))
#cv2.imwrite('../img.jpg', cv2.cvtColor(segmented[0],cv2.COLOR_RGB2BGR))
#cv2.imwrite('../img_vid.jpg', cv2.cvtColor(video[:,:,0:3],cv2.COLOR_RGB2BGR))
FPS = 25
secs = 3
writer = cv2.VideoWriter('layouttestvideo.avi', cv2.VideoWriter_fourcc(*'PIM1'), FPS, (128,128), True)
for i in range(75):
    frame = video[i,:,:,:].astype('uint8')
    for e in range(mus.shape[0]):
        Ploc = Plocs[e]
        mu = mus[e]
        track = Trackings[e]
        x0, y0 = unravel_index(Ploc[:,:,i//10].argmax(), Ploc[:,:,i//10].shape)
        x1 = int(x0 -  64*(mu[2*i//10]))
        y1 = int(y0 -  64*(mu[2*i//10+1]))
        x2 = int(x0 +  64*(mu[2*i//10]))
        y2 = int(y0 +  64*(mu[2*i//10+1]))
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        frame = cv2.rectangle(frame, (int(track[i][0]), int(track[i][1])), (int(track[i][2]), int(track[i][3])), (0, 255, 0), 1)

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cmap = 'gray', interpolation = 'bilinear')
        plt.show()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    writer.write(image)
writer.release()


a, b = load()

# -
'''

