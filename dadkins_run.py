import os
import numpy as np
import tensorflow as tf
import numpy as np
from dadkins_model import SentimentModel
from preprocess import *
import sys


def train(model, train_reviews, train_scores):
    batches = len(train_reviews) / model.batch_size
    for i in range(int(batches)):
        with tf.GradientTape() as tape:
            review_batch = tf.convert_to_tensor(train_reviews[i * model.batch_size : (i + 1) * model.batch_size])
            score_batch = train_scores[i * model.batch_size : (i + 1) * model.batch_size]

            call_result = model.call(review_batch)
            print("Call result: ", call_result.shape)
            loss = model.loss_function(call_result, score_batch)
            print(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_reviews, test_scores):
	batches = len(test_reviews) / model.batch_size

	loss = 0.0
	accuracy = 0.0
	num_correct_words = 0
	total_num_non_padding_words = 0

	for i in range(int(batches)):
		# slice arrays by batch size
		review_batch = test_reviews[i * model.batch_size : (i + 1) * model.batch_size]
		score_batch = test_scores[i * model.batch_size : (i + 1) * model.batch_size]

		# mask = english_batch[:, 1:] != eng_padding_index
        #
		# mask= tf.convert_to_tensor(mask, dtype=tf.float32)

		call_result = model.call(review_batch, score_batch)
		loss += model.loss_function(call_result, score_batch)

		accuracy = model.accuracy_function(call_result, score_batch)

	return np.exp(loss / int(batches)), accuracy


def main():
    print("Running preprocessing...")
    train_reviews, test_reviews, train_scores, test_scores, reviews_vocab = get_data()
    print("Preprocessing complete.")
    print("REVIEW VOCAB: ", len(reviews_vocab))
    print(len(train_reviews))

    model_args = (len(reviews_vocab))
    model = SentimentModel(batch_size=10, input_vocab_size=len(reviews_vocab))

    print("Training model.")
    train(model, train_reviews, train_scores)
    print("Training complete.")
    loss, acc = test(model, test_reviews, test_scores)
    print(loss, acc)

if __name__ == '__main__':
   main()
