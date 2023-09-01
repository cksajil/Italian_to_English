import tensorflow as tf
from os.path import join
from src import load_config
from src import encoder_decoder
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

config = load_config("config.yaml")
loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def custom_loss(real, pred):
    mask_loss = tf.math.logical_not(tf.math.equal(real, 0))
    loss_metric = loss_object(real, pred)

    mask_loss = tf.cast(mask_loss, dtype=loss_metric.dtype)
    loss_metric *= mask_loss
    return tf.reduce_mean(loss_metric)


def create_model():
    input_len = 20
    output_len = 20
    enc_units = 256
    att_units = 256
    dec_units = 256
    embedding_size = 100
    vocab_size_eng = 13416
    vocab_size_ita = 27029
    BATCH_SIZE = 512

    tf.keras.backend.clear_session()

    score_function = "dot"

    model_2_dot = encoder_decoder(
        encoder_inputs_length=input_len,
        decoder_inputs_length=output_len,
        output_vocab_size=vocab_size_eng,
        encoding_lang_vocab_size=vocab_size_ita + 1,
        decoding_lang_vocab_size=vocab_size_eng + 1,
        enc_embedding_size=embedding_size,
        dec_embedding_size=embedding_size,
        encoder_lstm_unit=enc_units,
        decoder_lstm_unit=dec_units,
        batch_size=BATCH_SIZE,
        score_fun=score_function,
        attn_units=att_units,
    )

    return model_2_dot


def load_pretrained_model():
    """
    Function to load pretrained attention model and return it
    """
    model = create_model()
    file_path = join(config["model_loc"], config["model_name"])
    model = load_model(file_path, custom_objects={"loss_function": custom_loss})
    return model
