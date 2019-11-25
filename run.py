import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys

## IMPORTANT!!!!!! TAKE OUT ID2WORD!!!@!
def train(model, train_french, train_english, eng_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initilized model to use for forward and backward pass
	:param train_french: french train data (all data for training) of shape (num_sentences, 14)
	:param train_english: english train data (all data for training) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
	:return: None
	"""

	# NOTE: For each training step, you should pass in the french sentences to be used by the encoder,
	# and english sentences to be used by the decoder
	# - The english sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP]
	#
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP]


	for i in range(0, len(train_french)//model.batch_size):
		sliced_fr = train_french[i * model.batch_size:model.batch_size*(i+1)]
		sliced_en = train_english[i * model.batch_size :model.batch_size*(i+1)]


		with tf.GradientTape() as tape:
			predictions = model(sliced_fr, sliced_en[:,:-1])
			decoded_symbols = tf.argmax(input=predictions, axis=2)


			mask = tf.not_equal(sliced_en[:,1:], eng_padding_index)
			loss = model.loss_function(predictions, sliced_en[:,1:], mask)
			acc = model.accuracy_function(predictions, sliced_en[:, 1:], mask)
			if i % 10 == 0:

				print("Step", i, ":", loss.numpy(), ",", acc.numpy(), "%")

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	pass

def test(model, test_french, test_english, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initilized model to use for forward and backward pass
	:param test_french: french test data (all data for testing) of shape (num_sentences, 14)
	:param test_english: english test data (all data for testing) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
	:returns: perplexity of the test set, per symbol accuracy on test set
	"""

	loss = 0
	acc = 0
	for i in range(0, len(test_french)//model.batch_size):
		sliced_fr = test_french[i * model.batch_size:model.batch_size*(i+1)]
		sliced_en = test_english[i * model.batch_size :model.batch_size*(i+1)]



		predictions = model(sliced_fr, sliced_en[:,:-1])
		mask = tf.not_equal(sliced_en[:,1:], eng_padding_index)
		loss += model.loss_function(predictions, sliced_en[:,1:], mask)
		acc += model.accuracy_function(predictions, sliced_en[:, 1:], mask)
	# Note: Follow the same procedure as in train() to construct batches of data!
	return (tf.exp(loss/(len(test_french)//model.batch_size))).numpy(),(acc/(len(test_french)//model.batch_size)).numpy()

def main():
	if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
			print("USAGE: python assignment.py <Model Type>")
			print("<Model Type>: [RNN/TRANSFORMER]")
			exit()

	print("Running preprocessing...")
	train_english,test_english, train_french,test_french, english_vocab,french_vocab,eng_padding_index = get_data('data/fls.txt','data/els.txt','data/flt.txt','data/elt.txt')
	print("Preprocessing complete.")

	model_args = (FRENCH_WINDOW_SIZE,len(french_vocab),ENGLISH_WINDOW_SIZE, len(english_vocab))
	if sys.argv[1] == "RNN":
		model = RNN_Seq2Seq(*model_args)
	elif sys.argv[1] == "TRANSFORMER":
		model = Transformer_Seq2Seq(*model_args)

	id2word = {v: k for k, v in english_vocab.items()}

	# TODO:
	# Train and Test Model for 1 epoch.
	train(model, train_french, train_english, eng_padding_index)
	loss, acc = test(model, test_french, test_english, eng_padding_index)
	print(loss, acc)

if __name__ == '__main__':
   main()
