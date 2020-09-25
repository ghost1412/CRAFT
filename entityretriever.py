from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import plot_model
import numpy as np

class EntityRetriever():

	def __init__(self, video_width=128, video_height=128, gt_directory=None, video_directory, annotations=None, F=75, LSTM=None):
		self.video_width = video_width
		self.video_height = video_height
		self.gt_directory = gt_directory
		self.video_directory = video_directory
		self.F = F
		self.annotations = annotations
		self.LSTM = LSTM
		self.model = None

	def CNN(self):
		CNN1 = Input((self.video_height, self.video_width, 3*self.F), name='Vi_1')
                CNN2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='
relu')(CNN1)
                (CNN2) = Dropout(0.9)(CNN1)
                CNN3 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation=
'relu')(CNN2)
                (CNN3) = Dropout(0.9)(CNN2)
                CNN4 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', activation=
'relu')(CNN3)
                (CNN4) = Dropout(0.9)(CNN3)
                CNN5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='same', activation='relu')(CNN4)
		return Model(inputs=CNN1, outputs=CNN5)

	def createModel(self):
		In = []
		cnn = []
		CNN = []
		for i in range(self.F):
			cnn.append(self.cnn())
		for i in range(self.F):
			In.append(Input((128, 128, 1)))
			In.append(cnn[i].input)
			CNN.append(cnn[i].output)
		
		self.model = Model(inputs=In, outputs=CNN)
		self.model.summary()
		plot_model(model, to_file='QUERYEMBEDDING.png', show_shapes=True)

	def fit(self, batch_size=32):
		



entityretriever = Entityretriever()
entityretireiver.fit(batch_size=32)
