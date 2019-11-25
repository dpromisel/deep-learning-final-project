import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, input_window_size, input_vocab_size, english_window_size):

		super(Transformer_Seq2Seq, self).__init__()

		self.review_vocab_size = input_vocab_size # The size of vocab from input reviews (preprocess.py)
		self.score_size = 5 # The range of possible amazon customer reviews

		self.review_window_size = review_window_size # The review window size

		self.batch_size = 100
		self.embedding_size = 100 # CHANGE

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


		# Define english and french embedding layers:
		self.review_embedding = tf.Variable(tf.random.truncated_normal(shape=[self.review_vocab_size, self.embedding_size], stddev=0.1, dtype=tf.float32))
		self.score_embedding = tf.Variable(tf.random.truncated_normal(shape=[self.score_size, self.embedding_size], stddev=0.1, dtype=tf.float32))

		# Create positional encoder layer for reviews
		self.review_pos_embedding = transformer.Position_Encoding_Layer(self.review_window_size, self.embedding_size)

		# Define encoder and decoder layers:
		self.encoder = transformer.Transformer_Block(self.embedding_size, False, False)
		self.decoder = transformer.Transformer_Block(self.embedding_size, True, False)

		# Define dense layer(s)
		self.dense1 = tf.keras.layers.Dense(units=self.embedding_size, activation=tf.nn.relu)
		self.dense2 = tf.keras.layers.Dense(units=self.score_size, activation=tf.nn.softmax)


	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to reviews
		:param decoder_input: batched scores
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x score_size]
		"""

		encode_embedding = tf.nn.embedding_lookup(self.review_embedding, encoder_input)
		decode_embedding = tf.nn.embedding_lookup(self.score_embedding, decoder_input)

		review_pos_embedding = self.review_pos_embedding(encode_embedding)
		encoding = self.encoder(review_pos_embedding)

		decoding = self.decoder(decode_embedding,encoding)

		dense1 = self.dense1(decoding)
		dense2 = self.dense2(dense1)

		return dense2

	def accuracy_function(self, prbs, labels, mask):
		"""
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x score_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the model cross-entropy loss after one forward pass

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x score_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		# Note: you can reuse this from rnn_model.
		loss = tf.reduce_mean(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs),mask))
		return loss
