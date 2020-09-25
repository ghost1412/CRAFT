from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import plot_model
from keras import backend as K
from cnn import *
from attention_layer import AttentionLayers
from lstm import lstm

F = 8


def bilinear_kernel(h, w, channels, use_bias = True, dtype = "float32") :

	y = np.zeros((h,w,channels,channels), dtype = dtype)
	for i in range(0, h):
		for j in range(0, w):
			y[i,j,:,:] = np.identity(channels) / float(h*w*1)
	if use_bias : return [y,np.array([0.], dtype = dtype)]
	else : return [y]


def channelPool(x):
	return K.max(x,axis=-1)

def channelAvg(x):
	return K.mean(x,axis=-1)

def buildModel():
	
	CNN = cnnModel(F)
	LSTM = Input((100,))
	# CONCATINATE WITH LSTM
	repli = RepeatVector(32)(LSTM)
	repli = Reshape((32*100,))(repli)
	repli = RepeatVector(32)(repli)
	repli = Reshape((32,32,100))(repli)	

	merged = concatenate([CNN.output, repli])
	backbone = Conv2D(2, kernel_size=(1, 1), padding='same', activation='relu')(merged)

	# FULLY CONV. LOCATION MLP
	Ploc1 = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu')(merged) 
	Ploc2 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu')(Ploc1)
	Ploc3 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu')(Ploc2)
	Ploc4 = Conv2D(F, kernel_size=(1, 1), padding='same', activation='relu')(Ploc3)

	# UPSAMPLING FOR LOCATION OUTPUT
	Ploc = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(Ploc4)
	
	#TODO
	#Ploc = Conv2D(filters = F, kernel_size = (4, 4), strides=(1,1), 
	#	activation = 'softmax', padding = 'same', use_bias = False,
	#	weights = bilinear_kernel(4, 4, F, False))(Ploc4)
	
	# CHANNEL MAXPOOLING AND MERGE WITH CNN
	Max = Lambda(channelPool)(Ploc4)
	Avg = Lambda(channelAvg)(backbone)
	attention = AttentionLayer()([Avg, Max])
	#attention = Flatten()(attention)
	attention = Reshape((32*32,))(attention)
	# SCALE MLP
	mu1 = Dense(256, activation='relu')(attention)
	mu2 = Dense(128, activation='relu')(mu1)
	mu = Dense(2*F, activation='sigmoid')(mu2)

	# MODEL SUMMARY
	model = Model(inputs=[CNN.input, LSTM], outputs=[Ploc, mu])
	model.summary()
	plot_model(model, to_file='LAYOUTCOMPOSER.png', show_shapes=True)
	return model

def lossfn(y_true, y_pred):
	Ploc_true = y_true[0]
	mu_true = y_true[1]
	Ploc_pred = y_pred[0]
	mu_pred = y_pred[1]
	cost = tf.add(tf.add(tf.add(-tf.log(Ploc_pred) + 0.5 * tf.log(tf.matrix_determinant(sigma)) + 0.5 * tf.dot(tf.transpose(mu_true - mu_pred), tf.dot(tf.matrix_inverse(sigma), mu_true - mu_pred)) + tf.log(2 * m.pi))))
	return cost


def complileModel():
	model = buildModel()
	
buildModel()
