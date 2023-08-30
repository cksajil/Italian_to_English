import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from attention import Attention
from tensorflow.keras.layers import Dense


class OneStepDecoder(tf.keras.Model):
    def __init__(
        self,
        tar_vocab_size,
        embedding_dim,
        input_length,
        dec_units,
        score_fun,
        att_units,
    ):
        super(OneStepDecoder, self).__init__()
        self.target_vocab_size = tar_vocab_size
        self.embedding_dim = embedding_dim
        self.decoder_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        self.input_length = input_length

        self.embedding_layer = Embedding(
            input_dim=tar_vocab_size,
            output_dim=embedding_dim,
            input_length=input_length,
            name="embedding_layer_decoder",
        )

        self.lstm1 = LSTM(
            units=dec_units,
            return_sequences=True,
            return_state=True,
            name="Decoder_LSTM",
        )

        self.lstm2 = LSTM(
            units=dec_units,
            return_sequences=True,
            return_state=True,
            name="Decoder_LSTM",
        )

        self.attention = Attention(self.score_fun, self.att_units)

        self.fc = Dense(units=tar_vocab_size)

    def call(self, input_to_decoder, encoder_output, state_h, state_c):
        output_embedding = self.embedding_layer(input_to_decoder)
        decoder_output, decoder_state_h, decoder_state_c = self.lstm1(
            encoder_output, initial_state=[state_h, state_c]
        )
        context_vector, attention_weights = self.attention(
            decoder_state_h, encoder_output
        )
        combined_vector = tf.concat(
            [tf.expand_dims(context_vector, axis=1), output_embedding], axis=-1
        )
        decoder_output, decoder_h, decoder_c = self.lstm2(combined_vector)

        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
        output = self.fc(decoder_output)

        return output, decoder_h, decoder_c, attention_weights, context_vector
