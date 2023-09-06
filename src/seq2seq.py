import pickle
import numpy as np
import tensorflow as tf
from os.path import join
from src import load_pretrained_model
from .general import load_config
from tensorflow.keras.preprocessing.sequence import pad_sequences


config = load_config("config.yaml")


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

    tok_seq = ita_tokenizer.texts_to_sequences([text_input.lower()])
    padded_in = pad_sequences(tok_seq, maxlen=20, dtype="int32", padding="post")
    tokenized_intext = tf.convert_to_tensor(padded_in)

    h = tf.zeros([1, 256])
    c = tf.zeros([1, 256])
    initial_state = [h, c]

    encoder_output, encoder_h, encoder_c = model.layers[0](
        tokenized_intext, initial_state
    )

    start_token = eng_tokenizer.word_index["<start>"]
    end_token = eng_tokenizer.word_index["<end>"]
    init_states = [encoder_h, encoder_c]
    decoder_hidden_state = encoder_h
    decoder_cell_state = encoder_c

    curr_vec = np.array([start_token])
    curr_vec = np.reshape(curr_vec, (1, 1))

    DECODER_SEQ_LENGTH = 20
    prediction_string = []
    attention_plot = np.zeros((20, 20))

    for index in range(DECODER_SEQ_LENGTH):
        (
            output,
            decoder_hidden_state,
            decoder_cell_state,
            attention_weights,
            contex_vector,
        ) = model.layers[1].oneStepDecoder(
            curr_vec, encoder_output, decoder_hidden_state, decoder_cell_state
        )

        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[index] = attention_weights.numpy()

        pred_dec_indx = np.argmax(output)
        prediction_string.append(eng_tokenizer.index_word[pred_dec_indx])

        if pred_dec_indx == end_token:
            break
        curr_vec = np.array([pred_dec_indx])
        curr_vec = np.reshape(curr_vec, (1, 1))
    return " ".join(prediction_string[:-1]), attention_plot
