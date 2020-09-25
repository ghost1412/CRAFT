from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import plot_model
from RoiPooling import RoiPoolingConv
from keras import regularizers
import json
from loaddata import load
import numpy as np
from lstm import lstm
import tensorflow as tf
import keras.backend as K
from handler import DataGeneratorEntity
#from keras.bechend import slice

MAX_LEN = 75  # Max length of review (in words)

class EntityRetriever():

	def __init__(self, video_width=128, video_height=128, F=3, batch_size=32, alpha=0.1, gt_directory=None, video_directory=None, annotations=None, video_database=None):
		self.video_width = video_width
		self.video_height = video_height
		self.gt_directory = gt_directory
		self.video_directory = video_directory
		self.F = F
		self.batch_size = batch_size
		self.alpha = alpha
		self.annotations = annotations
		self.LSTM = LSTM
		self.video_database = video_database
		self.model = None


	def CNN(self):
		CNN1 = Input((self.video_height, self.video_width, 3))
		CNN2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu')(CNN1)
		CNN3 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu')(CNN2)
		CNN4 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', activation='relu')(CNN3)
		CNN5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='same', activation='relu')(CNN4)
		return Model(inputs=CNN1, outputs=CNN5)


	def triplet_loss(self, x):
		query = x[:, :, 0:128]
		target = x[:, :, 128:]
		loss_triplet = 0
		for i in range(self.batch_size):
			for j in range(self.batch_size):
				if(j!=i):
					#loss1 = tf.add(tf.subtract(tf.reduce_sum(tf.square(tf.subtract(query[i], target[j])), 1), tf.reduce_sum(tf.square(tf.subtract(query[i], target[i])), 1), self.alpha))
					#loss2 = tf.add(tf.subtract(tf.reduce_sum(tf.square(tf.subtract(query[j], target[i])), 1), tf.reduce_sum(tf.square(tf.subtract(query[i], target[i])), 1), self.alpha))
					loss1 = tf.keras.backend.maximum(float(0), tf.add(self.alpha, tf.subtract(K.sum(query[i] * target[j],axis=-1, keepdims=True), K.sum(query[i] * target[i],axis=-1,keepdims=True))))
					loss2 = tf.keras.backend.maximum(float(0), tf.add(self.alpha, tf.subtract(K.sum(query[j] * target[i],axis=-1, keepdims=True), K.sum(query[i] * target[i],axis=-1,keepdims=True))))
					loss_triplet = loss_triplet + loss1 + loss2 

  		return loss_triplet

	def createModel(self):

                # Time Distributed ROI Pooling workaround due to kears TimeDistributed layer input limitation
                merged_input = Input((32*32*512+4,))
                Vi = Lambda(lambda merged_input: merged_input[0:32*32*512], output_shape=(32*32*512, ))(merged_input)
                Vi = Reshape((32, 32, 512))(Vi)

                l = Lambda(lambda merged_input: merged_input[32*32*512:], output_shape=(4, ))(merged_input)
                l = Reshape((1, 4))(l)

                pooling = RoiPoolingConv(7, 1)([Vi, l])
                pooling = Reshape((7,7,512))(pooling)
                ROIPooling = Model(inputs=merged_input, outputs=pooling)
                ROIPooling.summary()

		# QUERRY EMBEDDING NETWORK
		
		# Predicted Location and Partially Constructed Video
		InLE = Input((self.F, 4))
		InVE = Input((self.F, self.video_height, self.video_width, 3))
		
		# Time Distributed CNN for each frame
		CNNE1 = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu'))(InVE)
		CNNE2 = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu'))(CNNE1)
		CNNE3 = TimeDistributed(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', activation='relu'))(CNNE2)
		CNNE4 = TimeDistributed(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='same', activation='relu'))(CNNE3)
		CNNE_Reshape = Reshape((self.F, 32*32*512))(CNNE4)
		ROI_InputE = Concatenate(axis=2)([CNNE_Reshape, InLE])

		# Time Distributed RoIPooling
		ROI_CNNE = TimeDistributed(ROIPooling)(ROI_InputE)

		# Roi Pooling without TimeDistributed 
		#pooling = [Reshape((1,7,7,512))(RoiPoolingConv(7, 1)([Lambda(lambda z: z[:,x])(CNN4), Reshape((1, 4))(Lambda(lambda z: z[:,x])(InL))])) for x in range(self.F)]
		#ROI_CNN = Concatenate(axis=1)(pooling)
		ROI_CNNE = TimeDistributed(GlobalAveragePooling2D())(ROI_CNNE)		

		# Bidirectional LSTM for each frame
		LSTME = Bidirectional(LSTM(64, return_sequences=True, return_state=True), merge_mode='concat')(ROI_CNNE)
		LSTME = Reshape((self.F, 128))(LSTME[0])

		# Text Encodings input and concatenation with LSTM output
		InT = Input((128,))
		Intr = RepeatVector(self.F)(InT)	
		concat = Concatenate(axis=2)([LSTME, Intr])

		# MLP for final score
		MLP = TimeDistributed(Dense(256, activation='relu'))(concat)
		MLP = TimeDistributed(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))(MLP)

		# TARGET EMBEDDING NETWORK

		# Candidate Entity location and Video Frames
		InLT = Input((self.F, 4))
		InVT = Input((self.F, self.video_height, self.video_width, 3))

		# Time Distributed CNN for each frame of Target video
		CNNT1 = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu'))(InVT)
                CNNT2 = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu'))(CNNT1)
                CNNT3 = TimeDistributed(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', activation='relu'))(CNNT2)
                CNNT4 = TimeDistributed(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='same', activation='relu'))(CNNT3)
                CNNT_Reshape = Reshape((self.F, 32*32*512))(CNNT4)
                ROI_InputT = Concatenate(axis=2)([CNNT_Reshape, InLT])

                # Time Distributed ROI Pooling 
                ROI_CNNT = TimeDistributed(ROIPooling)(ROI_InputT)

                # Roi Pooling without TimeDistributed
                ROI_CNNT = TimeDistributed(GlobalAveragePooling2D())(ROI_CNNT)

                # Bidirectional LSTM for each frame
                LSTMT = Bidirectional(LSTM(64, return_sequences=True, return_state=True), merge_mode='concat')(ROI_CNNT)
                LSTMT = Reshape((self.F, 128))(LSTMT[0])

		#Triplet loss
		merged_output = concatenate([MLP, LSTMT])
		loss = Lambda(self.triplet_loss, (1, ))(merged_output)

                self.model = Model(inputs=[InLE, InVE, InT, InLT, InVT], outputs=loss)
		self.model.summary()
		opt = Adam(lr=0.001, decay=0.5, amsgrad=False)	#weight decay 0.0001
		self.model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['accuracy'])

		#plot_model(self.model, to_file='entityretriever_model.png', show_shapes=True)

	def fit(self, epochs, batch_size=32, verbose=1, workers=10):
		self.createModel()
		with open('layoutannotations.json') as f:
			annotations = json.load(f)
		_, split = load()
		with open('lstminput.json') as f:
			lstminput = json.load(f)
		training_generator = DataGeneratorEntity(annotations=annotations, video_files=split['train'], F=self.F, batch_size=batch_size, LSTM=self.LSTM, lstminput=lstminput)
		validation_generator = DataGeneratorEntity(annotations=annotations, video_files=split['val'], F=self.F, batch_size=batch_size, LSTM=self.LSTM, lstminput=lstminput)
		self.model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, use_multiprocessing=True, workers=workers, verbose=verbose)
	
		return

entityretriever = EntityRetriever()
entityretriever.fit(50)
#entityretireiver.fit(batch_size=32)from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import plot_model
from RoiPooling import RoiPoolingConv
from keras import regularizers
import json
from loaddata import load
import numpy as np
from lstm import lstm
import tensorflow as tf
import keras.backend as K
from handler import DataGeneratorEntity
#from keras.bechend import slice

MAX_LEN = 75  # Max length of review (in words)

class EntityRetriever():

	def __init__(self, video_width=128, video_height=128, F=3, batch_size=32, alpha=0.1, gt_directory=None, video_directory=None, annotations=None, video_database=None):
		self.video_width = video_width
		self.video_height = video_height
		self.gt_directory = gt_directory
		self.video_directory = video_directory
		self.F = F
		self.batch_size = batch_size
		self.alpha = alpha
		self.annotations = annotations
		self.LSTM = LSTM
		self.video_database = video_database
		self.model = None


	def CNN(self):
		CNN1 = Input((self.video_height, self.video_width, 3))
		CNN2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu')(CNN1)
		CNN3 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu')(CNN2)
		CNN4 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', activation='relu')(CNN3)
		CNN5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='same', activation='relu')(CNN4)
		return Model(inputs=CNN1, outputs=CNN5)


	def triplet_loss(self, x):
		query = x[:, :, 0:128]
		target = x[:, :, 128:]
		loss_triplet = 0
		for i in range(self.batch_size):
			for j in range(self.batch_size):
				if(j!=i):
					#loss1 = tf.add(tf.subtract(tf.reduce_sum(tf.square(tf.subtract(query[i], target[j])), 1), tf.reduce_sum(tf.square(tf.subtract(query[i], target[i])), 1), self.alpha))
					#loss2 = tf.add(tf.subtract(tf.reduce_sum(tf.square(tf.subtract(query[j], target[i])), 1), tf.reduce_sum(tf.square(tf.subtract(query[i], target[i])), 1), self.alpha))
					loss1 = tf.keras.backend.maximum(float(0), tf.add(self.alpha, tf.subtract(K.sum(query[i] * target[j],axis=-1, keepdims=True), K.sum(query[i] * target[i],axis=-1,keepdims=True))))
					loss2 = tf.keras.backend.maximum(float(0), tf.add(self.alpha, tf.subtract(K.sum(query[j] * target[i],axis=-1, keepdims=True), K.sum(query[i] * target[i],axis=-1,keepdims=True))))
					loss_triplet = loss_triplet + loss1 + loss2 

  		return loss_triplet

	def createModel(self):

                # Time Distributed ROI Pooling workaround due to kears TimeDistributed layer input limitation
                merged_input = Input((32*32*512+4,))
                Vi = Lambda(lambda merged_input: merged_input[0:32*32*512], output_shape=(32*32*512, ))(merged_input)
                Vi = Reshape((32, 32, 512))(Vi)

                l = Lambda(lambda merged_input: merged_input[32*32*512:], output_shape=(4, ))(merged_input)
                l = Reshape((1, 4))(l)

                pooling = RoiPoolingConv(7, 1)([Vi, l])
                pooling = Reshape((7,7,512))(pooling)
                ROIPooling = Model(inputs=merged_input, outputs=pooling)
                ROIPooling.summary()

		# QUERRY EMBEDDING NETWORK
		
		# Predicted Location and Partially Constructed Video
		InLE = Input((self.F, 4))
		InVE = Input((self.F, self.video_height, self.video_width, 3))
		
		# Time Distributed CNN for each frame
		CNNE1 = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu'))(InVE)
		CNNE2 = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu'))(CNNE1)
		CNNE3 = TimeDistributed(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', activation='relu'))(CNNE2)
		CNNE4 = TimeDistributed(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='same', activation='relu'))(CNNE3)
		CNNE_Reshape = Reshape((self.F, 32*32*512))(CNNE4)
		ROI_InputE = Concatenate(axis=2)([CNNE_Reshape, InLE])

		# Time Distributed RoIPooling
		ROI_CNNE = TimeDistributed(ROIPooling)(ROI_InputE)

		# Roi Pooling without TimeDistributed 
		#pooling = [Reshape((1,7,7,512))(RoiPoolingConv(7, 1)([Lambda(lambda z: z[:,x])(CNN4), Reshape((1, 4))(Lambda(lambda z: z[:,x])(InL))])) for x in range(self.F)]
		#ROI_CNN = Concatenate(axis=1)(pooling)
		ROI_CNNE = TimeDistributed(GlobalAveragePooling2D())(ROI_CNNE)		

		# Bidirectional LSTM for each frame
		LSTME = Bidirectional(LSTM(64, return_sequences=True, return_state=True), merge_mode='concat')(ROI_CNNE)
		LSTME = Reshape((self.F, 128))(LSTME[0])

		# Text Encodings input and concatenation with LSTM output
		InT = Input((128,))
		Intr = RepeatVector(self.F)(InT)	
		concat = Concatenate(axis=2)([LSTME, Intr])

		# MLP for final score
		MLP = TimeDistributed(Dense(256, activation='relu'))(concat)
		MLP = TimeDistributed(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))(MLP)

		# TARGET EMBEDDING NETWORK

		# Candidate Entity location and Video Frames
		InLT = Input((self.F, 4))
		InVT = Input((self.F, self.video_height, self.video_width, 3))

		# Time Distributed CNN for each frame of Target video
		CNNT1 = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu'))(InVT)
                CNNT2 = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu'))(CNNT1)
                CNNT3 = TimeDistributed(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', activation='relu'))(CNNT2)
                CNNT4 = TimeDistributed(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='same', activation='relu'))(CNNT3)
                CNNT_Reshape = Reshape((self.F, 32*32*512))(CNNT4)
                ROI_InputT = Concatenate(axis=2)([CNNT_Reshape, InLT])

                # Time Distributed ROI Pooling 
                ROI_CNNT = TimeDistributed(ROIPooling)(ROI_InputT)

                # Roi Pooling without TimeDistributed
                ROI_CNNT = TimeDistributed(GlobalAveragePooling2D())(ROI_CNNT)

                # Bidirectional LSTM for each frame
                LSTMT = Bidirectional(LSTM(64, return_sequences=True, return_state=True), merge_mode='concat')(ROI_CNNT)
                LSTMT = Reshape((self.F, 128))(LSTMT[0])

		#Triplet loss
		merged_output = concatenate([MLP, LSTMT])
		loss = Lambda(self.triplet_loss, (1, ))(merged_output)

                self.model = Model(inputs=[InLE, InVE, InT, InLT, InVT], outputs=loss)
		self.model.summary()
		opt = Adam(lr=0.001, decay=0.5, amsgrad=False)	#weight decay 0.0001
		self.model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['accuracy'])

		plot_model(self.model, to_file='entityretriever_model.png', show_shapes=True)
		print("sdd")

	def fit(self, epochs, batch_size=32, verbose=1, workers=10):
		self.createModel()
		with open('layoutannotations.json') as f:
			annotations = json.load(f)
		_, split = load()
		with open('lstminput.json') as f:
			lstminput = json.load(f)
		training_generator = DataGeneratorEntity(annotations=annotations, video_files=split['train'], F=self.F, batch_size=batch_size, LSTM=self.LSTM, lstminput=lstminput)
		validation_generator = DataGeneratorEntity(annotations=annotations, video_files=split['val'], F=self.F, batch_size=batch_size, LSTM=self.LSTM, lstminput=lstminput)
		self.model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, use_multiprocessing=True, workers=workers, verbose=verbose)
	
		return

entityretriever = EntityRetriever()
entityretriever.createModel()
#entityretireiver.fit(batch_size=32)
