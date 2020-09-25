from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import plot_model
from keras.optimizers import Adam
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
from livelossplot import PlotLossesKeras
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
			gt_directory = 'flintstones_dataset/layoutcomposer_gt/'
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


	def lossfn(self,y_true, y_pred):
		#TODO
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
		likelihood = K.tf.distributions.Bernoulli(logits=y_pred)
		return - K.sum(likelihood.log_prob(y_true), axis=-1)

	def loss2(self, y_true, y_pred):
		sigma = K.cast_to_floatx(np.diag(np.full((2*self.F,), 0.005)))
		return (0.5) * K.dot(K.dot((y_true- y_pred), tf.matrix_inverse(sigma)), tf.transpose(y_true- y_pred)) + self.F * tf.log(2*m.pi) - 6.903089986991*0.5
		#return self.F * (0.5) * tf.log(tf.matrix_determinant(sigma)) + (0.5) * K.dot(K.dot((y_true- y_pred), tf.matrix_inverse(sigma)), tf.transpose(y_true- y_pred))  + self.F * tf.log(2*m.pi)

	def createModel(self):
		Vi_1 = Input((self.video_height, self.video_width, 3*self.F), name='Vi_1')
		CNN1 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu')(Vi_1)
		CNN1 = BatchNormalization()(CNN1)
		CNN2 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu')(CNN1)
		CNN2 = BatchNormalization()(CNN2)
		CNN3 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', activation='relu')(CNN2)
		CNN3 = BatchNormalization()(CNN3)
		CNN = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='same', activation='relu')(CNN3)
		CNN = BatchNormalization()(CNN)
		Ei = Input((100,), name='Ei')
		
		# CONCATINATE WITH Embeddings
		repli = RepeatVector(32)(Ei)
		repli = Reshape((32*100,))(repli)
		repli = RepeatVector(32)(repli)
		repli = Reshape((32,32,100))(repli)	
	
		merged = concatenate([CNN, repli])
		#2D Grid coordinates
		#TODO

		# FULLY CONV. LOCATION MLP

		Ploc1 = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu')(CNN) 
		Ploc1 = BatchNormalization()(Ploc1)
		Ploc2 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu')(Ploc1)
		Ploc2 = BatchNormalization()(Ploc2)
		Ploc3 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu')(Ploc2)
		Ploc4 = Conv2D(self.F, kernel_size=(1, 1), padding='same', activation='relu')(Ploc2)
		Ploc4 = BatchNormalization()(Ploc4)
		# UPSAMPLING FOR LOCATION OUTPUT
		Ploc = UpSampling2D(size=(4, 4), data_format=None, interpolation='bilinear')(Ploc4)
		Ploc = Activation('sigmoid')(Ploc)
	
		# CHANNEL MAXPOOLING AND MERGE WITH CNN
		Max = Lambda(self.channelPool)(Ploc4)
		AttentionMap = Reshape((32*32,))(Max)
		AttentionMap = RepeatVector(612)(AttentionMap)
		AttentionMap = Reshape((32,32,612))(AttentionMap)		
		attention = multiply([merged, AttentionMap])

		attention = GlobalMaxPooling2D(data_format='channels_last')(attention)
		#attention = AttentionLayer()([Avg, Max])

		#attention = Flatten()(attention)
		#attention = Reshape((32*32,))(attention)

		# SCALE MLP
		mu1 = Dense(256, activation='relu')(attention)
		mu2 = Dense(128, activation='relu')(mu1)
		mu = Dense(2*self.F, activation='sigmoid')(mu2)

		# MODEL SUMMARY
		self.model = Model(inputs=[Vi_1, Ei], outputs=[mu, Ploc])
		self.model.summary()
		plot_model(self.model, to_file='layoutcomposer_model.png', show_shapes=True)
		
		# COMPILE MODEL
		opt = Adam(lr=0.001, decay=0.5, amsgrad=False)	#weight decay 0.0001
		self.model.compile(optimizer=opt, loss={'dense_3':self.loss2, 'activation_1':self.loss1}, loss_weights=[1, 1], metrics=['accuracy'])


	def fit(self, epochs, batch_size=32, verbose=1,workers=10):

			self.createModel()
			with open('layoutannotations.json') as f:
				annotations = json.load(f)
			_, split = load()
			with open('lstminput.json') as f:
				lstminput = json.load(f)
			tf.flags.DEFINE_integer("batch_size", 32, "Batch size during training")
			tf.flags.DEFINE_integer("eval_batch_size", 8, "Batch size during evaluation")

			training_generator = DataGeneratorLayout(annotations=annotations, video_files=split['train'], F=self.F, batch_size=batch_size, LSTM=self.LSTM, lstminput=lstminput, graph=self.graph)
			validation_generator = DataGeneratorLayout(annotations=annotations, video_files=split['val'], F=self.F, batch_size=batch_size, LSTM=self.LSTM, lstminput=lstminput, graph=self.graph)
			self.model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, use_multiprocessing=True, workers=workers, verbose=verbose, callbacks=[PlotLossesKeras()])
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
		with open('layoutTrainHistory.json', 'w') as fp:
			json.dump(self.model.history, fp)

#layoutcomposer = LayoutComposer(F=3)
#layoutcomposer.fit(50,64)
#layoutcomposer.save_history()

