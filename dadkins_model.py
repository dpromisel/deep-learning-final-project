import tensorflow as tf
from tensorflow.keras.layers import Dense

class MHAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size=100, num_heads=1):
        super(MHAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        assert self.hidden_size % self.num_heads == 0

        self.depth = self.hidden_size // self.num_heads

        self.wq = tf.keras.layers.Dense(self.hidden_size)
        self.wk = tf.keras.layers.Dense(self.hidden_size)
        self.wv = tf.keras.layers.Dense(self.hidden_size)

        self.dense = tf.keras.layers.Dense(self.hidden_size)
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        return output, attention_weights
    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, None)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.hidden_size))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class TBlock(tf.keras.Model):
    def __init__(self, input_vocab_size, hidden_size=100, num_heads=1):
        super(TBlock,self).__init__()
        self.Attn = MHAttention(num_heads=num_heads)

        self.Norm1 = tf.keras.layers.LayerNormalization()


        self.Dense1 = Dense(units=hidden_size, activation=tf.keras.activations.relu)
        self.Dense2 = Dense(units=hidden_size)

        self.Norm2 = tf.keras.layers.LayerNormalization()
        # TODO: add dropout

    def call(self, inputs):
        print("Generating attention")
        attention = self.Attn(inputs,inputs,inputs)[0]
        print("Attention: ", attention)
        # TODO: add layer normalization
        # norm1 = self.Norm1(attention + inputs)
        # print("Norm1: ", norm1)

        dense1 = self.Dense1(attention)
        print("Dense1: ", dense1)

        dense2 = self.Dense2(dense1)
        print("Dense2: ", dense2)

        return dense2


class TransformerModel(tf.keras.Model):
    def __init__(self,input_vocab_size,hidden_size=100,num_heads=1,net_size=1):
        super(TransformerModel,self).__init__()
        self.blocks = tf.keras.Sequential()
        for i in range(0, net_size):
            self.blocks.add(TBlock(input_vocab_size=input_vocab_size, hidden_size=hidden_size, num_heads=num_heads))

    def call(self, inputs):
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
        print(inputs.shape)
        inputs = tf.reshape(inputs, [-1])
        embedded = tf.nn.embedding_lookup(self.WordEmbedding, inputs)
        pos_embedding = self.Pos(embedded)
        print("Pos_embedding: ", pos_embedding)
        encoded = self.Transformer(pos_embedding)
        classes = self.Classification(encoded)
        return classes

    def loss_function(self, prbs, labels):
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs))
        return loss
