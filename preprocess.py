import numpy as np
import tensorflow as tf
import numpy as np
import csv
import nltk
import collections
from nltk.tokenize import RegexpTokenizer

##########DO NOT CHANGE#####################
REVIEW_FILE = "./data/amazon_camera_reviews.tsv"
SAMPLE_FILE = "./data/sample_us.tsv"
##########DO NOT CHANGE#####################

def build_vocab(sentences):
	"""
	DO NOT CHANGE

	Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
	"""
	tokens = []
	for s in sentences:
		tokens.extend(s.split())

	all_words = sorted(list(set(tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab

def build_inputs(reviews, token_set):
	inputs = []

	for review in reviews:
		review_tokens = [w for w in review.split() if w in token_set]
		inputs.append(" ".join(review_tokens))
	
	return inputs

def clean_reviews(raw_reviews):
	ratings = []
	reviews = []
	tokenizer = RegexpTokenizer(r'\w+')

	for review in raw_reviews:
		ratings.append(review["star_rating"])
		review_text = review["review_headline"].lower() + " " + review["review_body"].lower()
		review_tokens = tokenizer.tokenize(review_text)
		reviews.append(" ".join(review_tokens))

	return reviews, ratings

def build_token_set(reviews, num_common=5000):
	
	tokens = []
	for review in reviews:
		tokens.extend(review.split())
	
	most_common = collections.Counter(tokens).most_common(num_common)
	token_set = set([word[0] for word in most_common])
	
	return token_set

def read_data(file_name):
	"""
	DO NOT CHANGE

  	Load text data from file

	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  	"""
	reviews = []
	with open(file_name, 'rt') as data_file:
		reader = csv.DictReader(data_file, dialect='excel-tab')
		for row in reader:
			reviews.append(row)

	return reviews


def get_data():
	"""
	"""

	test_fraction = 0.1
	
	#1) Read Review Data for training and testing (see read_data)
	raw_reviews = read_data(SAMPLE_FILE)

	#2) Clean all reviews (remove punctutaion and convert to lower case)
	reviews, labels = clean_reviews(raw_reviews)

	#3) Consolidate most common words from the training reviews into single set
	split = int(len(reviews) * (1 - test_fraction))
	train_reviews = reviews[:split]
	token_set = build_token_set(train_reviews)

	#4) Create inputs and labels
	inputs = build_inputs(reviews, token_set)

	#5) Build Reviews Vocab
	review_vocab = build_vocab(token_set)

	#6) Split into test and train data
	train_inputs, test_inputs = inputs[:split], inputs[split:]
	train_labels, test_labels = labels[:split], labels[split:]

	return train_inputs, test_inputs, train_labels, test_labels, review_vocab
