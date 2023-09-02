import os
import yaml
import requests
from tqdm import tqdm
import tensorflow as tf
from os.path import join
from .models import encoder_decoder
from tensorflow.keras.losses import SparseCategoricalCrossentropy


loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def load_config(config_name):
    """
    A function to load and return config file in YAML format
    """
    CONFIG_PATH = "./config/"
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("config.yaml")


def create_models_folder():
    """Function to create a folder in a location if it does not exist"""
    if not os.path.exists(config["model_loc"]):
        os.makedirs(config["model_loc"])


def download_model():
    """
    Download pretrained models if not already downloaded
    """
    if not os.path.exists(os.path.join(config["model_loc"], config["model_zip_name"])):
        print("Downloading pretrained models if not exists")
        doi = config["model_doi"]
        response = requests.get(f"https://zenodo.org/api/records/{doi}")
        data = response.json()
        files = data["files"]
        formats = ".hdf5"
        model_files = [file for file in files if file["key"].endswith(formats)]

        if len(model_files) == 0:
            print("No model files found.")
        else:
            model_file = model_files[0]
            file_url = model_file["links"]["self"]
            model_filename = model_file["key"]

            response = requests.get(file_url, stream=True)
            segments = response.iter_content()
            with open(os.path.join(config["model_loc"], model_filename), "wb") as file:
                for chunk in tqdm(segments):
                    file.write(chunk)

            print(f"Model downloaded as {model_filename}")


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
    filepath = join(config["model_loc"], config["model_weights_file_name"])
    model.load_weights(filepath, skip_mismatch=False, by_name=False, options=None)
    return model
