from loaddata import*

import tensorflow as tf
import keras
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MAX_LEN = 75  # Max length of review (in words)

def preProcessData():
	annotation, trainTest = load()
	data = []
	words = []
	tags = []
	sentences = []
	sentences_tag = []

	for i in range(len(annotation)):
		data.append(annotation[i]["parse"]["pos_tags"])

	for i in range(len(annotation)):
		sentences.append([])
		sentences_tag.append([])
		for j in range(len(annotation[i]["parse"]["pos_tags"])):
			words.append(annotation[i]["parse"]["pos_tags"][j][0])	
			sentences[i].append(annotation[i]["parse"]["pos_tags"][j][0])
			sentences_tag[i].append(annotation[i]["parse"]["pos_tags"][j][1])
			tags.append(annotation[i]["parse"]["pos_tags"][j][1])
	words = list(set(words))
	tags = list(set(tags))

	word2idx = {w: i + 2 for i, w in enumerate(words)}
	word2idx["UNK"] = 1 # Unknown words
	word2idx["PAD"] = 0 # Padding
	
	idx2word = {i: w for w, i in word2idx.items()}

	tag2idx = {t: i+1 for i, t in enumerate(tags)}
	tag2idx["PAD"] = 0

	idx2tag = {i: w for w, i in tag2idx.items()}
	
	#print("Barney walks into the dining room and takes an apple out of a pig's mouth. The pig wakes up and speaks to him.: {}".format(word2idx["Barney"]))
	with open('word2idx.json','w') as f:
		json.dump(word2idx,f)
	print("Saved word2indx.json")
	

	from keras.preprocessing.sequence import pad_sequences

	# Convert each sentence from list of Token to list of word_index
	X = []
	y = []
	lstminput = dict()
	for i in range(len(sentences)):
		X.append([])
		for j in range(len(sentences[i])):
			X[i].append(word2idx[sentences[i][j]])

	X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2idx["PAD"])

	for i in range(len(sentences)):
		lstminput.update({annotation[i]['globalID']:np.array(X[i]).tolist()})

	with open('lstminput.json','w') as f:
		json.dump(lstminput,f)	
	print("Saved idx of each sentence in annotations")

	for i in range(len(sentences)):
		y.append([])
		for j in range(len(sentences_tag[i])):
			y[i].append(tag2idx[sentences_tag[i][j]])

	y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["PAD"])
	print(len(tags))
	from keras.utils import to_categorical
	# One-Hot encode
	y = [to_categorical(i, num_classes=len(tags)+1) for i in y]  # n_tags+1(PAD)


	from sklearn.model_selection import train_test_split
	X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
	X_tr.shape, X_te.shape, np.array(y_tr).shape, np.array(y_te).shape
	return X_tr, X_te, y_tr, y_te, len(words), len(tags), idx2tag, words



