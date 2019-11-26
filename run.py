import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
# from transformer_model import Transformer_Seq2Seq
# from rnn_model import RNN_Seq2Seq
import sys

def train(model, train_reviews, train_scores):
	pass

def test(model, test_reviews, test_scores):
	pass

def main():
	print("Running preprocessing...")
	train_scores,test_scores, train_reviews,test_reviews, scores_vocab, reviews_vocab = get_data()
	print("Preprocessing complete.")

	# model = Transformer_Seq2Seq(*model_args)

	# id2word = {v: k for k, v in scores_vocab.items()}

	# print("Training model.")
	# train(model, train_reviews, train_scores, eng_padding_index)
	# print("Training complete.")
	# loss, acc = test(model, test_reviews, test_scores, eng_padding_index)
	# print(loss, acc)

if __name__ == '__main__':
   main()
