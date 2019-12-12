import numpy as np
import tensorflow as tf
import numpy as np
import csv
import nltk
import collections
from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk.corpus import stopwords
##########DO NOT CHANGE#####################
REVIEW_FILE = "./data/amazon_camera_reviews.tsv"
SAMPLE_FILE = "./data/sample_us.tsv"

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
##########DO NOT CHANGE#####################


def remove_stop_word(reviews):
	# nltk.download('stopwords')
	# nltk.download('punkt')
	stop_words = set(stopwords.words('english'))
	reviews_no_stop = []
	for review in reviews:
		word_tokens = word_tokenize(review)
		str = ""
		for w in word_tokens:
			if not w in stop_words:
				str+=w
		reviews_no_stop.append(str)
	return reviews_no_stop

def pad_corpus(reviews, max_length = 50):
	"""
	DO NOT CHANGE:

	arguments are lists of REVIEW, ENGLISH sentences. Returns [REVIEW-sents, ENGLISH-sents]. The
	text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
	the end.

	:param review: list of reviews
	:return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
	"""
	REVIEW_padded_sentences = []
	for review in reviews:
		review = review.split()
<<<<<<< HEAD
		padded_REVIEW = review[:REVIEW_WINDOW_SIZE-1]
		padded_REVIEW += [STOP_TOKEN] + [PAD_TOKEN] * (REVIEW_WINDOW_SIZE - len(padded_REVIEW)-1)
		REVIEW_padded_sentences.append(" ".join(padded_REVIEW))
=======
		padded_REVIEW = review[:max_length]
		padded_REVIEW += [STOP_TOKEN] + [PAD_TOKEN] * (max_length - len(padded_REVIEW)-1)
		padded_REVIEW = " ".join(padded_REVIEW)

		REVIEW_padded_sentences.append(padded_REVIEW)
>>>>>>> 1c066f5ac270b704ef506a41e07d4adc339c226e

	return REVIEW_padded_sentences

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

	vocab =  {word:i for i, word in enumerate(all_words)}

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

def word_to_id(reviews, word2id, max_length = 50):

	reviews_ids = []

	for review in reviews:
		ids = list(map(lambda x: word2id[x], review.split()))
		if (len(ids) > max_length):
			ids = ids[:max_length]
		if len(ids) < max_length:
			ids += [word2id[PAD_TOKEN]] * (max_length - len(ids))
		reviews_ids.append(ids)

	return reviews_ids

def get_data(sample=True, max_length=50):
	"""
	"""

	test_fraction = 0.1

	#1) Read Review Data for training and testing (see read_data)
	if (sample):
		raw_reviews = read_data(SAMPLE_FILE)
	else:
		raw_reviews = read_data(REVIEW_FILE)

	#2) Clean all reviews (remove punctutaion and convert to lower case)
	reviews, labels = clean_reviews(raw_reviews)
	# print(type(reviews))
	# reviews = remove_stop_word(reviews)
	# print(type(reviews))
	reviews = pad_corpus(reviews, max_length=max_length)

	#3) Consolidate most common words from the training reviews into single set
	token_set = build_token_set(reviews)

	#4) Create inputs and labels
	inputs = build_inputs(reviews, token_set)
	#5) Build Word to Id and Id to Word dictionaries
	id2word = {i: w for i, w in enumerate(list(token_set))}
	word2id = {w: i for i, w in enumerate(list(token_set))}

	label_nums = [int(w) for i, w in enumerate(labels)]

	#6) Split into test and train data
	split = int(len(reviews) * (1 - test_fraction))

	import random
	c = list(zip(inputs, label_nums))

	random.shuffle(c)

	inputs, label_nums = zip(*c)

	train_words, test_words = inputs[:split], inputs[split:]
	train_labels, test_labels = label_nums[:split], label_nums[split:]
	#7) Convert training and testing set from list of words to list of IDs
	train_ids = word_to_id(train_words, word2id, max_length=max_length)
	test_ids = word_to_id(test_words, word2id, max_length=max_length)
	# total = 0
	# for id in test_ids:
	# 	pad_id = word2id[PAD_TOKEN]
	# 	stop_id = word2id[STOP_TOKEN]
	# 	if id != pad_id and id != stop_id:
	# 		total+=1
	# print(total)
	return train_ids, test_ids, train_labels, test_labels, word2id, id2word
