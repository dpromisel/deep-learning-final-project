import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from model import SentimentModel
from pylab import *
import sys

from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Input, Dropout, concatenate, Layer, InputSpec, LSTM
from keras import activations, initializers, regularizers, constraints

from transformer_funcs import Transformer_Block, Position_Encoding_Layer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def visualize_data(title_, values, x_label = "none", y_label="none"):
	x_values = range(0, len(values), 1)
	y_values = values
	plot(x_values, y_values)
	xlabel(x_label)
	ylabel(y_label)
	title(title_)
	grid(True)
	show()

def train(model, train_reviews, train_scores, id2word):
	batches = len(train_reviews) / model.batch_size
	print("NUM_BATCHES: ", batches)
	# print(train_reviews)
	losses = []
	for i in range(int(batches)):
		with tf.GradientTape() as tape:

			# slice arrays by batch size
			review_batch = train_reviews[i * model.batch_size : (i + 1) * model.batch_size]
			score_batch = train_scores[i * model.batch_size : (i + 1) * model.batch_size]

			call_result = model.call(tf.convert_to_tensor(review_batch))

			loss = model.loss_function(tf.convert_to_tensor(call_result, dtype=np.float32), (tf.convert_to_tensor(np.array(score_batch)>3, dtype=tf.float32)))
			losses.append(loss)
			accuracy = model.accuracy_function(np.array(call_result)>0.5, np.array(score_batch)>3)
			if (i % 50 == 0):
				print(i, "| Loss: ", loss.numpy(), "| Acc: ", accuracy)
				# if ((call_result[0] < 0.5 and score_batch[0] > 3) or (call_result[0] > 0.5 and score_batch[0] < 3)):
				# 	print("Misclassified sample: ", call_result[0].numpy(), " ".join(list(map(lambda x: id2word[x], review_batch[0]))))

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	return losses

def test(model, test_reviews, test_scores, id2word):
	batches = len(test_reviews) / model.batch_size

	accs = []
	misclassified_as_good = {}
	misclassified_as_bad = {}
	for i in range(int(batches)):
		review_batch = test_reviews[i * model.batch_size : (i + 1) * model.batch_size]
		score_batch = test_scores[i * model.batch_size : (i + 1) * model.batch_size]
		call_result = model.call(tf.convert_to_tensor(review_batch))

		for j in range(0, len(call_result)):
			if (call_result[j] < 0.5 and score_batch[j] > 3):
				for word in (list(map(lambda x: id2word[x], review_batch[j]))):
					if word in misclassified_as_bad:
						misclassified_as_bad[word] = misclassified_as_bad[word]+1
					else:
						misclassified_as_bad[word] = 1
			if (call_result[j] > 0.5 and score_batch[j] < 3):
				for word in (list(map(lambda x: id2word[x], review_batch[j]))):
					if word in misclassified_as_good:
						misclassified_as_good[word] = misclassified_as_good[word]+1
					else:
						misclassified_as_good[word] = 1
		accuracy = model.accuracy_function(np.array(call_result)>0.5, np.array(score_batch)>3)
		# print("Accuracy batch ", i, " | acc:", accuracy)

		accs.append(accuracy)
		# words = list(map(lambda x: id2word[x], review_batch[0]))
		# print(words, call_result[0])
	print("Misclassified as bad:")
	print_sort(misclassified_as_bad)
	print("Misclassified as good:")
	print_sort(misclassified_as_good)
	return accs

def print_sort(dict):
	if (len(dict) > 0):
		for key, value in sorted(dict.items(), key=lambda item: item[1])[::-1]:
			if (not key == "*PAD*" and not key == "*STOP*"):
				print("%s: %s" % (key, value))
	else:
		print("none")

def TransformerSentimentModel(num_words=5000, max_length = 50, hidden_size = 256, embedding_size=64):
	embedding_matrix = np.random.normal(0, 0.2, (num_words, embedding_size))
	inputs = Input(shape=(max_length, ))
	embedded = Embedding(num_words, embedding_size, weights=[embedding_matrix], trainable=True)(inputs)
	dropout1 = Dropout(0.1)(embedded)
	pos_encoding = Position_Encoding_Layer(max_length, embedding_size)(dropout1)
	transformed = Transformer_Block(hidden_size, False)(pos_encoding)
	dropout2 = Dropout(0.1)(tf.reduce_mean(transformed, axis=1))
	dense1 = Dense(hidden_size, activation="relu")(dropout2)
	dropout3 = Dropout(0.1)(dense1)
	dense2 = Dense(1, activation="sigmoid")(dropout3) ## Output layer!
	model = Model(inputs=inputs, outputs=dense2)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	return model

def LSTMSentimentModel(num_words=5000, max_length = 50, hidden_size = 256, embedding_size=64):
	embedding_matrix = np.random.normal(0, 0.2, (num_words, embedding_size))
	inputs = Input(shape=(max_length, ))
	embedded = Embedding(num_words, embedding_size, weights=[embedding_matrix], trainable=True)(inputs)
	dropout1 = Dropout(0.1)(embedded)
	lstm1 = LSTM(hidden_size)(dropout1)
	lstm2 = LSTM(hidden_size)(dropout1)
	concatenated = concatenate([lstm1, lstm2])
	dropout2 = Dropout(0.1)(concatenated)
	dense1 = Dense(embedding_size, activation="relu")(dropout2)
	dropout3 = Dropout(0.1)(dense1)
	dense2 = Dense(1, activation="sigmoid")(dropout3) ## Output layer!
	model = Model(inputs=inputs, outputs=dense2)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	return model

def main():
	## CUSTOM HYPERPARAMETERS
	max_length = 50

	sample = "sample" in sys.argv
	lstm = "lstm" in sys.argv

	if (sample):
		batch_size = 10
	else:
		batch_size = 1000
	print("Initializing " + ("lstm" if lstm else "transformer") + " model on " + ("sample" if sample else "full") + " dataset.")

	print("Running preprocessing. This could take up to 5min.")
	train_reviews, test_reviews, train_scores, test_scores, reviews_vocab, id2word = get_data(sample=sample, max_length=50)
	print("Preprocessing complete.")

	print("Good reviews: ", len(np.nonzero(np.array(train_scores)>3)[0]), "Bad reviews: ", len(np.nonzero(np.array(train_scores)<=3)[0]), "Total: ", len(train_scores))

	if (lstm):
		model = LSTMSentimentModel(num_words=len(reviews_vocab), max_length=max_length)
	else:
		model = TransformerSentimentModel(num_words=len(reviews_vocab), max_length=max_length)

	print("REVIEW VOCAB LENGTH: ", len(reviews_vocab))
	history = model.fit(np.array(train_reviews), np.array(train_scores)>3, batch_size=1000, epochs=5, shuffle = True, validation_split=0.30)
	score, acc = model.evaluate(np.array(test_reviews), np.array(test_scores)>3, batch_size=10)
	print('Test score:', score)
	print('Test accuracy:', acc)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()
	# print("Training model:")
	#
	# num_epochs = 10
	# losses = []
	# for i in range(0, num_epochs):
	# 	epoch_losses = train(model, train_reviews, train_scores, id2word)
	# 	losses.extend(epoch_losses)
	#
	# print("Training complete.")

	# accs = test(model, test_reviews, test_scores, id2word)
	# print("Final accuracy: ", np.mean(accs))

	# visualize_data("Losses", losses, "training iteration", "loss")


if __name__ == '__main__':
   main()
