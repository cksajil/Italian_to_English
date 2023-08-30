import tensorflow as tf
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Dense


class Attention(tf.keras.layers.Layer):
    def __init__(self, scoring_function, att_units):
        super(Attention, self).__init__()
        self.att_units = att_units
        self.scoring_function = scoring_function

        if self.scoring_function == "dot":
            pass
        if scoring_function == "general":
            self.fc = Dense(units=att_units)
        elif scoring_function == "concat":
            self.WE = Dense(units=att_units)
            self.WD = Dense(units=att_units)
            self.v = Dense(units=1)

    def call(self, decoder_hidden_state, encoder_output):
        decoder_hidden_state = tf.expand_dims(decoder_hidden_state, axis=1)

        if self.scoring_function == "dot":
            similarity = Dot(axes=(2, 2))([encoder_output, decoder_hidden_state])

        elif self.scoring_function == "general":
            weighted_encoder_output = self.fc(encoder_output)
            similarity = Dot(axes=(2, 2))(
                [weighted_encoder_output, decoder_hidden_state]
            )

        elif self.scoring_function == "concat":
            weighted_encoder = self.WE(encoder_output)
            weighted_decoder = self.WD(decoder_hidden_state)
            tan_h_activation = tf.nn.tanh(weighted_decoder + weighted_encoder)
            similarity = self.v(tan_h_activation)

        attention_weights = tf.nn.softmax(similarity, axis=1)
        context = attention_weights * encoder_output
        context = tf.reduce_sum(context, axis=1)
        return context, attention_weights
