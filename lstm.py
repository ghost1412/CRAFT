from preprocess import preProcessData
from keras.models import Model, Input
from keras.layers import *
from keras_contrib.layers import CRF
from keras.utils import plot_model
from keras.models import load_model
import numpy as np
#from livelossplot import PlotLossesKeras

BATCH_SIZE = 512  # Number of examples used in each iteration
EPOCHS = 12  # Number of passes through entire dataset
MAX_LEN = 75  # Max length of review (in words)
EMBEDDING = 40  # Dimension of word embedding vector


def lstm():
	n_words = 6414
	n_tags=42
	input = Input(shape=(MAX_LEN,))
	model = Embedding(input_dim=n_words+2, output_dim=EMBEDDING, # n_words + 2 (PAD & UNK)
		          input_length=MAX_LEN, mask_zero=True)(input)  # default: 20-dim embedding
	model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
	model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
	#model, forword_h, forword_c, backword_h, backword_c  = Bidirectional(LSTM(units=50, return_sequences=True, return_state=True,
		                   #recurrent_dropout=0.1))(model)  # variational biLSTM
	#state_h = Concatenate()([forword_h, backword_h])
	#state_c = Concatenate()([forword_c, backword_c])
	model = TimeDistributed(Dense(100, activation="relu"))(model)  # a dense layer as suggested by neuralNer
	crf = CRF(n_tags+1)  # CRF layer, n_tags+1(PAD)
	out = crf(model)  # output
	model = Model(input, out)
	model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
	plot_model(model, to_file='lstm.png', show_shapes=True)
	model.summary()

	return model

if __name__ == "__main__":

	X_tr, X_te, y_tr, y_te, n_words, n_tags, idx2tag, words = preProcessData()
	print("n_words:{}, n_tags:{}".format(n_words,n_tags))
	model = lstm()
	history = model.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=0.1, callbacks=[PlotLossesKeras()])
	#model.save('lstm.h5')	
	

	# Eval
	pred_cat = model.predict(X_te)
	pred = np.argmax(pred_cat, axis=-1)
	y_te_true = np.argmax(y_te, -1)


	from sklearn_crfsuite.metrics import flat_classification_report

	# Convert the index to tag
	pred_tag = [[idx2tag[i] for i in row] for row in pred]
	y_te_true_tag = [[idx2tag[i] for i in row] for row in y_te_true] 

	report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
	print(report)

	

	i = np.random.randint(0, X_te.shape[0]) # choose a random number between 0 and len(X_te)
	p = model.predict(np.array([X_te[i]]))
	p = np.argmax(p, axis=-1)
	true = np.argmax(y_te[i], -1)

	print("Sample number {} of {} (Test Set)".format(i, X_te.shape[0]))
	# Visualization
	print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
	print(30 * "=")
	for w, t, pred in zip(X_te[i], true, p[0]):
		if w != 0:
			print("{:15}: {:5} {}".format(words[w-2], idx2tag[t], idx2tag[pred]))

	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()





