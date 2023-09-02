from tensorflow.keras import Model
from .encoder import Encoder
from .decoder import Decoder


class encoder_decoder(Model):
    def __init__(
        self,
        encoder_inputs_length,
        decoder_inputs_length,
        output_vocab_size,
        encoding_lang_vocab_size,
        decoding_lang_vocab_size,
        enc_embedding_size,
        dec_embedding_size,
        encoder_lstm_unit,
        decoder_lstm_unit,
        batch_size,
        score_fun,
        attn_units,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.encoder = Encoder(
            inp_vocab_size=encoding_lang_vocab_size,
            embedding_size=enc_embedding_size,
            enc_units=encoder_lstm_unit,
            input_length=encoder_inputs_length,
        )

        self.decoder = Decoder(
            out_vocab_size=decoding_lang_vocab_size,
            embedding_dim=dec_embedding_size,
            input_length=decoder_inputs_length,
            dec_units=decoder_lstm_unit,
            score_fun=score_fun,
            att_units=attn_units,
        )

    def call(self, data):
        encoder_input, decoder_input = data[0], data[1]
        encoder_output, encoder_h, encoder_c = self.encoder(
            encoder_input, self.encoder.initialize_states(self.batch_size)
        )
        decoder_h = encoder_h
        decoder_c = encoder_c
        decoder_output = self.decoder(
            decoder_input, encoder_output, decoder_h, decoder_c
        )

        return decoder_output
