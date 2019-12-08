import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class SentimentModelLSTM(tf.keras.Model):
	def __init__(self, input_vocab_size):

		super(SentimentModelLSTM, self).__init__()

		self.input_vocab_size = input_vocab_size # The size of vocab from input reviews (preprocess.py)
		self.score_size = 5 # The range of possible amazon customer reviews

		# REMOVE WINDOW SIZE!
		self.review_window_size = 20 # The review window size

		self.batch_size = 20
		self.embedding_size = 64 # CHANGE

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.embedding = tf.Variable(tf.random.truncated_normal(shape=[self.input_vocab_size, self.embedding_size], stddev=0.1, dtype=tf.float32))
		self.bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.embedding_size))
		self.dense1 = tf.keras.layers.Dense(self.embedding_size, activation='relu')
		self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

	@tf.function
	def call(self, inputs):
		"""
		:param encoder_input: batched ids corresponding to reviews
		:param decoder_input: batched scores
		:return prbs: The 2d probabilities as a tensor, [batch_size x score size]
		"""
		embedding = tf.nn.embedding_lookup(self.embedding, inputs)
		bidirectional = self.bidirectional(embedding)
		dense1 = self.dense1(bidirectional)
		dense2 = self.dense2(dense1)
		return dense2

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
