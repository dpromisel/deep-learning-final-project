import numpy as np
import tensorflow as tf
import numpy as np
import csv

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
	
	print(tokens)

	all_words = sorted(list(set(tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab

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
	reviews = read_data(SAMPLE_FILE)

	inputs = []
	labels = []

	#2) Grab labels and inputs from reviews
	for review in reviews:
		labels.append(review["star_rating"])
		inputs.append(review["review_headline"] + " " + review["review_body"])

	#3) Build Reviews Vocab
	review_vocab = build_vocab(inputs)
	# print(review_vocab)

	#4) Split into Test and Train data
	split = int(len(inputs) * (1 - test_fraction))
	train_inputs, test_inputs = inputs[:split], inputs[split:]
	train_labels, test_labels = labels[:split], labels[split:]

	# print("Train Size: ", len(train_inputs), len(train_labels))
	# print("Test Size: ", len(test_inputs), len(test_labels))

	return train_inputs, test_inputs, train_labels, test_labels, review_vocab
	