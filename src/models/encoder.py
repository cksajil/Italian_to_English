import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding


class Encoder(Model):
    def __init__(self, inp_vocab_size, embedding_size, enc_units, input_length):
        super(Encoder, self).__init__()
        self.inp_vocab_size = inp_vocab_size
        self.emedding_dim = embedding_size
        self.encoder_units = enc_units
        self.input_length = input_length

        self.embedding_layer = Embedding(
            input_dim=inp_vocab_size,
            output_dim=embedding_size,
            input_length=input_length,
            name="embedding_layer_encoder",
        )

        self.lstm = LSTM(
            units=enc_units,
            return_sequences=True,
            return_state=True,
            name="Encoder_LSTM",
        )

    def call(self, input_sequence, states):
        input_embeddings = self.embedding_layer(input_sequence)
        self.encoder_output, self.encoder_h, self.encoder_c = self.lstm(
            input_embeddings, initial_state=[states[0], states[1]]
        )
        return self.encoder_output, self.encoder_h, self.encoder_c

    def initialize_states(self, batch_size):
        self.init_h = tf.zeros((batch_size, self.encoder_units))
        self.init_c = tf.zeros((batch_size, self.encoder_units))
        return self.init_h, self.init_c
