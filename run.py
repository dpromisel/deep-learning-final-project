import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from model import SentimentModel
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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



			loss = model.loss_function(tf.convert_to_tensor(call_result, dtype=np.float32), (tf.convert_to_tensor(score_batch, dtype=np.float32)-1)/4)
			accuracy = model.accuracy_function(np.array(call_result)>0.5, np.array(score_batch)>3)
			if (i % 50 == 0):
				print(i, "| Loss: ", loss.numpy(), "| Acc: ", accuracy)
				# if ((call_result[0] < 0.5 and score_batch[0] > 3) or (call_result[0] > 0.5 and score_batch[0] < 3)):
				# 	print("Misclassified sample: ", call_result[0].numpy(), " ".join(list(map(lambda x: id2word[x], review_batch[0]))))

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_reviews, test_scores, id2word):
	batches = len(test_reviews) / model.batch_size

	accs = []
	misclassified_as_good = []
	misclassified_as_bad = []
	for i in range(int(batches)):
		review_batch = test_reviews[i * model.batch_size : (i + 1) * model.batch_size]
		score_batch = test_scores[i * model.batch_size : (i + 1) * model.batch_size]
		call_result = model.call(tf.convert_to_tensor(review_batch))
		if (call_result[0] < 0.5 and score_batch[0] > 3):
			for word in (list(map(lambda x: id2word[x], review_batch[0]))):
				if word in misclassified_as_bad:
					misclassified_as_bad[word] = misclassified_as_bad[word]+1
				else:
					misclassified_as_bad[word] = 1
		if (call_result[0] > 0.5 and score_batch[0] < 3):
			for word in (list(map(lambda x: id2word[x], review_batch[0]))):
				if word in misclassified_as_good:
					misclassified_as_good[word] = misclassified_as_good[word]+1
				else:
					misclassified_as_good[word] = 1
		accuracy = model.accuracy_function(np.array(call_result)>0.5, np.array(score_batch)>3)
		accs.append(accuracy)
		# words = list(map(lambda x: id2word[x], review_batch[0]))
		# print(words, call_result[0])
	print("Misclassified as bad:")
	print(misclassified_as_bad)
	print("Misclassified as good:")
	print(misclassified_as_good)
	return np.mean(accs)

def main():
	sample = "sample" in sys.argv
	lstm = "lstm" in sys.argv
	print("Initializing " + ("lstm" if lstm else "transformer") + " model on " + ("sample" if sample else "full") + " dataset.")

	print("Running preprocessing. This could take up to 5min.")
	train_reviews, test_reviews, train_scores, test_scores, reviews_vocab, id2word = get_data(sample=sample)
	print("Preprocessing complete.")

	print("REVIEW VOCAB LENGTH: ", len(reviews_vocab))

	model = SentimentModel(len(reviews_vocab), transformer=(not lstm), sample=sample)

	print("Training model:")

	train(model, train_reviews, train_scores, id2word)
	print("Training complete.")

	acc = test(model, test_reviews, test_scores, id2word)
	print("Final accuracy: ", acc)


if __name__ == '__main__':
   main()
