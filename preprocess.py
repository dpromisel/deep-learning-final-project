import numpy as np
import tensorflow as tf
import numpy as np
import csv

##########DO NOT CHANGE#####################
REVIEW_FILE = "./data/amazon_camera_reviews.tsv"
SAMPLE_FILE = "./data/sample_us.tsv"
REVIEW_WINDOW_SIZE = 14
##########DO NOT CHANGE#####################

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

	#1) Read Review Data for training and testing (see read_data)
	reviews = read_data(SAMPLE_FILE)

	inputs = []
	labels = []

	for review in reviews:
		labels.append(review["star_rating"])
		inputs.append(review["review_headline"] + " " + review["review_body"])

	print(inputs, labels)

	return inputs, labels, [], [], [], []
