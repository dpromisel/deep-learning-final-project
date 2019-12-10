import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from model import SentimentModel
import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train(model, train_reviews, train_scores, id2word):
	batches = len(train_reviews) / model.batch_size
	print("NUM_BATCHES: ", batches)
	# print(train_reviews)
	for i in range(int(batches)):
		with tf.GradientTape() as tape:

			# slice arrays by batch size
			review_batch = train_reviews[i * model.batch_size : (i + 1) * model.batch_size]
			score_batch = train_scores[i * model.batch_size : (i + 1) * model.batch_size]

			call_result = model.call(tf.convert_to_tensor(review_batch))

			if (call_result[0] < 0.5 and score_batch[0] > 3):
				print(call_result[0])
				print(list(map(lambda x: id2word[x], review_batch[0])))

			loss = model.loss_function(tf.convert_to_tensor(call_result, dtype=np.float32), (tf.convert_to_tensor(score_batch, dtype=np.float32)-1)/4)
			accuracy = model.accuracy_function(np.array(call_result)>0.5, np.array(score_batch)>3)
			if (i % 50 == 0):
				print(i, loss, accuracy)


		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_reviews, test_scores, id2word):
	batches = len(test_reviews) / model.batch_size

	accs = []
	for i in range(int(batches)):
		review_batch = test_reviews[i * model.batch_size : (i + 1) * model.batch_size]
		score_batch = test_scores[i * model.batch_size : (i + 1) * model.batch_size]
		call_result = model.call(tf.convert_to_tensor(review_batch))
		accuracy = model.accuracy_function(np.array(call_result)>0.5, np.array(score_batch)>3)
		accs.append(accuracy)
		# words = list(map(lambda x: id2word[x], review_batch[0]))
		# print(words, call_result[0])
	return np.mean(accs)

def main():
	print("Running preprocessing...")
	train_reviews, test_reviews, train_scores, test_scores, reviews_vocab, id2word = get_data()
	print("Preprocessing complete.")

	print("REVIEW VOCAB LENGTH: ", len(reviews_vocab))

	model = SentimentModel(len(reviews_vocab), transformer=True)

	print("Training model.")

	train(model, train_reviews, train_scores, id2word)
	print("Training complete.")

	acc = test(model, test_reviews, test_scores, id2word)
	print(acc)


if __name__ == '__main__':
   main()
