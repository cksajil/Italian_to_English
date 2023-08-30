import tensorflow as tf
from tensorflow.keras import Model
from one_step_decoder import OneStepDecoder


class Decoder(Model):
    def __init__(
        self,
        out_vocab_size,
        embedding_dim,
        input_length,
        dec_units,
        score_fun,
        att_units,
    ):
        super(Decoder, self).__init__()
        self.out_vocav_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        self.oneStepDecoder = OneStepDecoder(
            tar_vocab_size=self.out_vocav_size,
            embedding_dim=self.embedding_dim,
            input_length=self.input_length,
            dec_units=self.dec_units,
            score_fun=self.score_fun,
            att_units=self.att_units,
        )

    def call(
        self, input_to_decoder, encoder_output, decoder_hidden_state, decoder_cell_state
    ):
        output_array = tf.TensorArray(
            tf.float32, size=self.input_length, name="output_array"
        )

        for tstamp in range(self.input_length):
            (
                output,
                decoder_hidden_state,
                decoder_cell_state,
                attention_weights,
                context_vector,
            ) = self.oneStepDecoder(
                input_to_decoder[:, tstamp : tstamp + 1],
                encoder_output,
                decoder_hidden_state,
                decoder_cell_state,
            )
            output_array = output_array.write(tstamp, output)
        output_array = tf.transpose(output_array.stack(), [1, 0, 2])
        return output_array
