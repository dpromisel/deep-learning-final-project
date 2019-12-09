import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from model import SentimentModelLSTM
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train(model, train_reviews, train_scores):
	batches = len(train_reviews) / model.batch_size
	print("NUM_BATCHES: ", batches)
	# print(train_reviews)
	for i in range(int(batches)):
		with tf.GradientTape() as tape:

			# slice arrays by batch size
			review_batch = train_reviews[i * model.batch_size : (i + 1) * model.batch_size]
			score_batch = train_scores[i * model.batch_size : (i + 1) * model.batch_size]

			# mask = english_batch[:, 1:] != eng_padding_index
			# mask= tf.convert_to_tensor(mask, dtype=tf.float32)
			call_result = model.call(tf.convert_to_tensor(review_batch))

			loss = model.loss_function(tf.convert_to_tensor(call_result, dtype=np.float32), (tf.convert_to_tensor(score_batch, dtype=np.float32)-1)/4)
			accuracy = model.accuracy_function(np.array(call_result)>0.5, np.array(score_batch)>3)

			print(i, loss, accuracy)

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_reviews, test_scores):
	call_result = model.call(test_reviews)
	accuracy = model.accuracy_function(np.array(call_result)>0.5, np.array(test_scores)>3)
	return accuracy

def main():
	print("Running preprocessing...")
	train_reviews, test_reviews, train_scores, test_scores, reviews_vocab = get_data()
	print("Preprocessing complete.")




	print("REVIEW VOCAB LENGTH: ", len(reviews_vocab))

	model = SentimentModelLSTM(len(reviews_vocab))


	# id2word = {v: k for k, v in scores_vocab.items()}

	print("Training model.")
	# print(train_reviews)
	# model.fit(np.array(train_reviews), (np.array(train_scores) > 3), validation_split=0.1, epochs = 3)

	train(model, train_reviews, train_scores)
	print("Training complete.")

	acc = test(model, test_reviews, test_scores)
	print(acc)

if __name__ == '__main__':
   main()
