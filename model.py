import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

		######vvv DO NOT CHANGE vvv##################
		super(Transformer_Seq2Seq, self).__init__()

		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters
		# 2) Define embeddings, encoder, decoder, and feed forward layers

		# Define batch size and optimizer/learning rate
		self.batch_size = 100
		self.embedding_size = 100

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


		# Define english and french embedding layers:
		self.FR_embedding = tf.Variable(tf.random.truncated_normal(shape=[self.french_vocab_size, self.embedding_size], stddev=0.1, dtype=tf.float32))
		self.EN_embedding = tf.Variable(tf.random.truncated_normal(shape=[self.english_vocab_size, self.embedding_size], stddev=0.1, dtype=tf.float32))

		# Create positional encoder layers
		self.FR_pos_embedding = transformer.Position_Encoding_Layer(self.french_window_size, self.embedding_size)
		self.EN_pos_embedding = transformer.Position_Encoding_Layer(self.english_window_size, self.embedding_size)

		# Define encoder and decoder layers:
		self.encoder = transformer.Transformer_Block(self.embedding_size, False, False)
		self.decoder = transformer.Transformer_Block(self.embedding_size, True, False)

		# Define dense layer(s)
		self.dense1 = tf.keras.layers.Dense(units=self.embedding_size, activation=tf.nn.relu)
		self.dense2 = tf.keras.layers.Dense(units=self.english_vocab_size, activation=tf.nn.softmax)


	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""

		# TODO:
		#1) Add the positional embeddings to french sentence embeddings
		encode_embedding = tf.nn.embedding_lookup(self.FR_embedding, encoder_input)
		decode_embedding = tf.nn.embedding_lookup(self.EN_embedding, decoder_input)

		FR_pos_embedding = self.FR_pos_embedding(encode_embedding)

		#2) Pass the french sentence embeddings to the encoder
		encoding = self.encoder(FR_pos_embedding)

		#3) Add positional embeddings to the english sentence embeddings
		EN_pos_embedding = self.EN_pos_embedding(decode_embedding)

		#4) Pass the english embeddings and output of your encoder, to the decoder
		decoding = self.decoder(EN_pos_embedding,encoding)

		#3) Apply dense layer(s) to the decoder out to generate probabilities
		dense1 = self.dense1(decoding)
		dense2 = self.dense2(dense1)

		return dense2

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE

		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
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

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		# Note: you can reuse this from rnn_model.
		loss = tf.reduce_mean(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs),mask))
		return loss
