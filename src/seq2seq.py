import pickle
import tensorflow as tf
import numpy as np
from os.path import join
from src import load_config
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import SparseCategoricalCrossentropy


config = load_config("config.yaml")
loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def custom_loss(real, pred):
    mask_loss = tf.math.logical_not(tf.math.equal(real, 0))
    loss_metric = loss_object(real, pred)

    mask_loss = tf.cast(mask_loss, dtype=loss_metric.dtype)
    loss_metric *= mask_loss
    return tf.reduce_mean(loss_metric)


def load_pretrained_model():
    """
    Function to load pretrained attention model and return it
    """
    file_path = join(config["model_loc"], config["model_name"])
    model = load_model(file_path, custom_objects={"loss_function": custom_loss})
    return model


def load_tokenizer(language="italian"):
    """
    Function to load Italian and English text embedding tokenizers
    """
    if language == "italian":
        file_path = join(config["model_loc"], config["it_toknizr_model_name"])
    elif language == "english":
        file_path = join(config["model_loc"], config["en_toknizr_model_name"])
    with open(file_path, "rb") as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def predict(text_input):
    """
    Function to do model prediction from Italian to English
    """
    model = load_pretrained_model()
    ita_tokenizer = load_tokenizer()
    eng_tokenizer = load_tokenizer(language="english")

    tok_seq = ita_tokenizer.texts_to_sequences([text_input])
    padded_in = pad_sequences(tok_seq, maxlen=20, dtype="int32", padding="post")

    init_state = model.layers[0].initialize_states(1)
    encoder_output, encoder_h, encoder_c = model.layers[0](padded_in, init_state)

    tar_word = np.zeros((1, 1))
    tar_word[0, 0] = 1

    encoder_state_values = [encoder_h, encoder_c]
    stop_condition = False
    txt_sent = ""
    k = 0

    while not stop_condition:
        out, state_dec_h, state_dec_c = model.layers[1](tar_word, encoder_state_values)
        encoder_state_values = [state_dec_h, state_dec_c]
        out = model.layers[2](out)
        out = np.argmax(out, -1)
        k += 1
        if k > 20 or out == eng_tokenizer.word_index["<end>"]:
            stop_condition = True
        else:
            txt_sent = txt_sent + " " + eng_tokenizer.index_word[int(out)]
        tar_word = out.reshape(1, 1)
    return txt_sent
