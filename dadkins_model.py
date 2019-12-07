import tensorflow as tf
from tensorflow.keras.layers import Dense

class MHAttention(tf.keras.Model):
    def __init__(self, input_vocab_size, hidden_size=100, num_heads=1):

        ## Initialize our model
        super(MHAttention, self).__init__()

        self.input_vocab_size = input_vocab_size
        self.num_heads = num_heads

        ## STATIC PARAMETERS
        self.hidden_size = hidden_size

        self.Q = Dense(units=self.hidden_size)
        self.K = Dense(units=self.hidden_size)
        self.V = Dense(units=self.hidden_size)


        self.Total = Dense(self.hidden_size)

    def call(self, Q, K, V):
        q = tf.transpose(tf.reshape(self.Q(Q), [Q.shape[0], Q.shape[1], self.num_heads, self.hidden_size/self.num_heads]), [1,2])
        k = tf.transpose(tf.reshape(self.K(K), [K.shape[0], K.shape[1], self.num_heads, self.hidden_size/self.num_heads]), [1,2])
        v = tf.transpose(tf.reshape(self.V(V), [V.shape[0], V.shape[1], self.num_heads, self.hidden_size/self.num_heads]), [1,2])

        unscaled = tf.matmul(q, tf.transpose(k, [2,3]))

        # TODO: maybe cast
        weights = tf.nn.softmax(unscaled / tf.math.sqrt(tf.convert_to_tensor([self.head_size * 1.0])))

        v_weight = tf.transpose(tf.matmul(weights, v), [1,2])

        # calculate total weight vectors
        total = self.Total(tf.tranpose(v_weight, [q.shape[0],q.shape[1],self.hidden_size]))

        return total

class TBlock(tf.keras.Model):
    def __init__(self, input_vocab_size, hidden_size=100, num_heads=1):
        super(TBlock,self).__init__()
        self.Attn = MHAttention(input_vocab_size=input_vocab_size,num_heads=num_heads)

        self.Norm1 = tf.keras.layers.LayerNormalization()


        self.Dense1 = Dense(units=hidden_size, activation=tf.keras.activations.relu)
        self.Dense2 = Dense(units=hidden_size)

        self.Norm2 = tf.keras.layers.LayerNormalization()
        # TODO: add dropout

    def call(self, inputs):
        attention = self.Attn(inputs,inputs,inputs)
        # TODO: add layer normalization
        norm1 = self.Norm1(attention + inputs)

        dense1 = self.Dense1(norm1)
        dense2 = self.Dense2(dense1)
        norm2 = self.Norm2(dense2 + inputs)
        return norm2


class TransformerModel(tf.keras.Model):
    def __init__(self,input_vocab_size,hidden_size=100,num_heads=1,net_size=1):
        super(TransformerModel,self).__init__()
        self.blocks = tf.keras.Sequential()
        for i in range(0, net_size):
            self.blocks.add(TBlock(input_vocab_size=input_vocab_size, hidden_size=hidden_size, num_heads=num_heads))

    def call(inputs):
        return self.blocks(inputs)

class PositionEncodingLayer(tf.keras.layers.Layer):
	def __init__(self, max_length, embedding_size):
		super(PositionEncodingLayer, self).__init__()
		self.positional_embeddings = self.add_weight("pos_embed",shape=[max_length, embedding_size])

	@tf.function
	def call(self, x):
		return x+self.positional_embeddings

class SentimentModel(tf.keras.Model):
    def __init__(self, batch_size=200, input_vocab_size=1000, embedding_size=100, max_length = 200):
        super(SentimentModel,self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.batch_size = batch_size
        self.WordEmbedding = tf.Variable(tf.random.truncated_normal(shape=[input_vocab_size, embedding_size], stddev=0.1, dtype=tf.float32))
        self.Pos = PositionEncodingLayer(max_length=max_length, embedding_size=embedding_size)
        self.Transformer = TransformerModel(input_vocab_size=input_vocab_size, hidden_size=100, num_heads=1, net_size=1)
        self.Classification = Dense(units=5, activation=tf.keras.activations.relu)


    def call(self, inputs):
        inputs = tf.transpose(inputs, [-1])
        embedded = self.WordEmbedding(inputs)
        pos_embedding = self.Pos(embedded)
        encoded = self.Transformer(pos_embedding)
        classes = self.Classification(encoded)
        return classes

    def loss_function(self, prbs, labels):
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs))
        return loss
