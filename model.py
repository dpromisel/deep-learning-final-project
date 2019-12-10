import numpy as np
import tensorflow as tf
from transformer_funcs import Transformer_Block


class SentimentModel(tf.keras.Model):
	def __init__(self, input_vocab_size, transformer=True, sample=True):

		super(SentimentModel, self).__init__()

		self.input_vocab_size = input_vocab_size # The size of vocab from input reviews (preprocess.py)

		if (sample):
			self.batch_size = 1
		else:
			self.batch_size = 1000
		self.embedding_size = 64 # CHANGE

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.embedding = tf.Variable(tf.random.truncated_normal(shape=[self.input_vocab_size, self.embedding_size], stddev=0.1, dtype=tf.float32))
		self.is_transformer = transformer
		if (self.is_transformer):
			self.transformer = Transformer_Block(self.embedding_size, False)
		else:
			self.bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.embedding_size))

		self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')

	@tf.function
	def call(self, inputs):
		"""
		:param encoder_input: batched ids corresponding to reviews
		:param decoder_input: batched scores
		:return prbs: The 2d probabilities as a tensor, [batch_size x score size]
		"""
		embedding = tf.nn.embedding_lookup(self.embedding, inputs)

		if (self.is_transformer):
			transformed = self.transformer(embedding)
			dense = self.dense3(tf.reduce_mean(bidirectional, axis=1))
		else:
			bidirectional = self.bidirectional(embedding)
			dense = self.dense3(bidirectional)

		return dense

	def accuracy_function(self, predictions, labels):
		"""
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x score_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		accuracy = np.mean(np.equal(predictions, labels))
		return accuracy


	def loss_function(self, prbs, labels):
		"""
		Calculates the model cross-entropy loss after one forward pass

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x score_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		loss = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy()(prbs, labels))
		return loss
