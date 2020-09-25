from keras.models import Sequential, Model
from keras.layers import *
	
def cnnModel(F):
	
	CNN1 = Input((128, 128, 3*F))
	CNN2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu')(CNN1)
	CNN3 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu')(CNN2)
	CNN4 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', activation='relu')(CNN3)
	CNN5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='same', activation='relu')(CNN4)
	model = Model(inputs=CNN1, outputs=CNN5)
	model.summary()
	return model
